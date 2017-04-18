#include <assert.h>

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ros/ros.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"

#include "average.hpp"
#define BACKWARD_HAS_DW 1
#include "backward.hpp"
#include "bag_video_reader.hpp"
#include "calibration_data.hpp"
#include "clique.hpp"
#include "cross_frame_processor.hpp"
#include "debug_renderer.hpp"
#include "direction_tracker.hpp"
#include "direction_target.hpp"
#include "fps_meter.hpp"
#include "frame_processor.hpp"
#include "kitti_video_reader.hpp"
#include "math3d.hpp"
#include "reprojection_estimator.hpp"
#include "rigid_estimator.hpp"
#include "timer.hpp"

using namespace std;

namespace e = Eigen;
namespace po = boost::program_options;

backward::SignalHandling sh;

typedef std::pair<Eigen::Quaterniond, Eigen::Vector3d> Transform;

std::vector<Transform> readTransforms(const std::string& filename) {
  std::ifstream f(filename);

  std::vector<Transform> res;

  while (!f.eof()) {
    double rot_m[9];
    double t_data[3];

    for (int i=0; i < 3; ++i) {
      for (int j=0; j < 3; ++j) {
        f >> rot_m[i*3 + j];
      }
      f >> t_data[i] >> std::skipws;
    }

    Eigen::Quaterniond rot{Eigen::Matrix3d(rot_m)};
    Eigen::Vector3d t(t_data);

    res.push_back(std::make_pair(rot, t));
  }

  return res;
}

int main(int argc, char** argv) {
  string video_file, direction_file, calib_file, mono_calib_file;
  string path_trace_file;
  int start;
  int debug;
  std::string kitti_basedir, kitti_dataset;
  int frame_count;

  po::options_description desc("Command line options");
  desc.add_options()
      ("video-file",
       po::value<string>(&video_file),
       "path to the video file")
      ("direction-file",
       po::value<string>(&direction_file),
       "path to the direction file")
      ("start-frame",
       po::value<int>(&start)->default_value(0),
       "start at this second")
      ("total-frames",
       po::value<int>(&frame_count)->default_value(0),
       "number of frames to process")
      ("stereo-calib-file",
       po::value<string>(&calib_file),
       "path to the calibration file")
      ("mono-calib-file",
       po::value<string>(&mono_calib_file)->default_value(
         "data/calib_phone.yml"),
       "path to the calibration file")
      ("path-trace-file",
       po::value<string>(&path_trace_file),
       "save calculated path to the file")
      ("kitti-basedir",
       po::value<string>(&kitti_basedir),
       "Kitti datasets directory")
      ("kitti-dataset-name",
       po::value<string>(&kitti_dataset),
       "Analyze a KITTI sequence")
      ("debug",
       po::value<int>(&debug)->default_value(0),
       "Debug mode: 0 - none, 1 - failed frames, 2 - every frame");
 
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  RawStereoCalibrationData raw_calib;

  std::unique_ptr<VideoReader> rdr;
  if (kitti_dataset.empty()) {
    if (calib_file.empty() || video_file.empty()) {
      std::cerr << "--calib_file and --video-file should be specified when "
        << "--kitti_dataset is not" << std::endl;
      exit(1);
    }
    raw_calib = RawStereoCalibrationData::read(calib_file);
    rdr.reset(new BagVideoReader(video_file, "image_raw"));
  } else {
    auto kitti_reader = new KittiVideoReader(
        kitti_basedir + "/sequences/" + kitti_dataset);
    rdr.reset(kitti_reader);
    raw_calib = RawStereoCalibrationData::readKitti(
        kitti_basedir + "/sequences/" + kitti_dataset + "/calib.txt", 
        kitti_reader->imgSize());
    raw_calib.write("/tmp/kitti-calib.yml");
  }

  auto scaled_calib = raw_calib.resize(0.5);

  ros::init(argc, argv, "camera_processor");
  ros::NodeHandle ros_node;
  ros::AsyncSpinner spinner(1);

  auto odometry_publisher = 
      ros_node.advertise<geometry_msgs::PoseWithCovarianceStamped>(
          "odometry", 10);

  spinner.start();


  StereoCalibrationData calib(scaled_calib);
  MonoCalibrationData mono_calibration(
    RawMonoCalibrationData::read(mono_calib_file),
    calib.Pl.colRange(0, 3),
    calib.raw.size);

  int src_frame_width = raw_calib.size.width;
  int src_frame_height = raw_calib.size.height;
  int scaled_frame_width = scaled_calib.size.width;
  int scaled_frame_height = scaled_calib.size.height;

  cv::Mat src_frame_mat;
  src_frame_mat.create(src_frame_height, src_frame_width*2, CV_8UC1);

  cv::Mat scaled_frame_mat;
  scaled_frame_mat.create(scaled_frame_height, scaled_frame_width*2, CV_8UC1);

  cv::Mat mono_frames[] = {
    scaled_frame_mat(cv::Range::all(), cv::Range(0, scaled_frame_width)),
    scaled_frame_mat(
        cv::Range::all(), cv::Range(scaled_frame_width, scaled_frame_width*2))
  };

  rdr->skip(start);
    
  int global_frame_index = start-1;

  FrameProcessorConfig config;
  config.descriptor_radius = 40;
  config.initial_threshold = 15;
  FrameProcessor frame_processor(calib, config);

  FrameData frame_data[2] = {
    FrameData(config.max_keypoint_count),
    FrameData(config.max_keypoint_count)
  };

  FrameDebugData frame_debug_data[2];

  CrossFrameDebugData cross_frame_debug_data;

  CrossFrameProcessorConfig cross_processor_config;
  CrossFrameProcessor cross_processor(calib, cross_processor_config);

  int frame_index = -1;

  std::vector<cv::Point2d> img_keypoints;

  e::Quaterniond cam_r(1, 0, 0, 0);
  e::Vector3d cam_t(0, 0, 0);
  std::vector<Transform> ground_truth;
  if (!kitti_dataset.empty()) {
    ground_truth = readTransforms(
        kitti_basedir + "/poses/" + kitti_dataset + ".txt");
  }

  std::ofstream trace;
  if (!path_trace_file.empty()) {
    trace.open(path_trace_file.c_str(), ios::out);
  }

  e::Quaterniond d_r, d_r_prev(1, 0, 0, 0);
  e::Vector3d d_t, d_t_prev(0, 0, 0);

  bool done = false;
  while (!done && (frame_count == 0 || frame_index + 1 < frame_count)) {
    Timer timer;

    if (!rdr->nextFrame(src_frame_mat)) {
      break;
    }

    cv::resize(
        src_frame_mat, scaled_frame_mat, 
        cv::Size(scaled_calib.size.width * 2, scaled_calib.size.height));

    frame_index++;
    global_frame_index++;
    
    std::cout << "Frame #" << global_frame_index << std::endl;

    timer.mark("read");

    int cur_index = frame_index % 2;
    int prev_index = 1 - (frame_index % 2);
    FrameData& cur_frame = frame_data[cur_index];
    FrameData& prev_frame = frame_data[prev_index];

    frame_processor.process(
        mono_frames, cur_frame, 
        debug ? &frame_debug_data[cur_index] : nullptr);

    timer.mark("process");
    
    e::Affine3d gt_d;
    if (!ground_truth.empty()) {
      auto p = ground_truth[global_frame_index - 1];
      auto g = ground_truth[global_frame_index];
      gt_d = (e::Translation3d(p.second) * p.first).inverse() * 
        e::Translation3d(g.second) * g.first;
    }
 
    bool ok = false;
 
    e::Matrix3d t_cov;
    ok = cross_processor.process(
        cur_frame, 
        d_r, d_t, 
        &t_cov, 
        debug ? &cross_frame_debug_data : nullptr);

    timer.mark("cross");

    if (ok) {
      double e2 = t_cov.trace();
      double sigma = sqrt(e2);
      double d = d_t.norm();
      if (3 * sigma > d * 0.2 && d > 0.1) {
        std::cout << "FAIL";
        std::cout << "T_cov = " << t_cov << std::endl; 
        std::cout << "t = " << d_t << std::endl;
        std::cout << "speed = " << d << std::endl;
        ok = false;

        cam_t += cam_r*d_t_prev;
        cam_r = cam_r*d_r_prev;
        
      } else {
        cam_t += cam_r*d_t;
        cam_r = cam_r*d_r;

        d_r_prev = d_r;
        d_t_prev = d_t;
      }
      cam_r.normalize();

      geometry_msgs::PoseWithCovarianceStamped msg;
      msg.header.stamp = ros::Time::now();
      msg.header.seq = frame_index;

      if (ok) {
        msg.header.frame_id = "";
        msg.pose.pose.position.x = d_t.x() / 1E+3;
        msg.pose.pose.position.y = d_t.y() / 1E+3;
        msg.pose.pose.position.z = d_t.z() / 1E+3;
        msg.pose.pose.orientation.x = d_r.x();
        msg.pose.pose.orientation.y = d_r.y();
        msg.pose.pose.orientation.z = d_r.z();
        msg.pose.pose.orientation.w = d_r.w();
      } else {
        msg.header.frame_id = "invalid";
      }

      odometry_publisher.publish(msg);

      /* auto ypr = rotToYawPitchRoll(cam_r) * 180.0 / M_PI; */
      /* std::cout << "yaw = " << ypr.x() */ 
      /*           << ", pitch = " << ypr.y() */ 
      /*           << ", roll = " << ypr.z() << endl; */
      /* std::cout << "T = " << cam_t.transpose() << endl; */ 
      /* std::cout << "T_cov = " << t_cov << std::endl; */

      /* if (!ground_truth.empty()) { */
      /*   auto gt = ground_truth[global_frame_index]; */
      /*   auto ypr_gt = rotToYawPitchRoll(gt.first); */
      /*   auto t_gt = gt.second; */
      /*   std::cout << "GT: yaw = " << ypr_gt.x() */ 
      /*             << ", pitch = " << ypr_gt.y() */ 
      /*             << ", roll = " << ypr_gt.z() << endl; */
      /*   std::cout << "GT: T = " << t_gt.transpose() * 1000 << endl; */ 
      /* } */
    }

    if (trace.is_open()) {
      Eigen::Affine3d t = Eigen::Translation3d(cam_t) * cam_r;

      for (int i=0; i < 3; ++i) {
        for (int j=0; j < 4; ++j) {
          trace << std::setprecision(10) << t.matrix().data()[j*4 + i] << " ";
        }
      }
      trace << std::endl;
    }
    std::cout << "Times: " << timer.str() << std::endl;

    if (frame_index > 0 && (debug == 2 || (debug == 1 && !ok))) {
      std::unique_ptr<DebugRenderer> renderer(DebugRenderer::create(
          calib,
          frame_data[cur_index],
          frame_debug_data[1 - cur_index],
          frame_debug_data[cur_index],
          cross_frame_debug_data,
          ground_truth.empty() ? nullptr : &gt_d,
          1920, 1080));
      if (!renderer->loop()) {
        break;
      }
    } else if (debug == 3) {
      cv::imshow("debug", mono_frames[0]);
      cv::waitKey(1);
    }
  }

  return 0;
}
