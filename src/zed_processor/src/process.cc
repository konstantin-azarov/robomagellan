#include <assert.h>

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

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
#include "math3d.hpp"
#include "reprojection_estimator.hpp"
#include "rigid_estimator.hpp"
#include "timer.hpp"

using namespace std;

namespace e = Eigen;
namespace po = boost::program_options;

backward::SignalHandling sh;

int main(int argc, char** argv) {
  string video_file, direction_file, calib_file, mono_calib_file;
  string path_trace_file;
  int start;
  bool debug;
  int frame_count;

  po::options_description desc("Command line options");
  desc.add_options()
      ("video-file",
       po::value<string>(&video_file)->required(),
       "path to the video file")
      ("direction-file",
       po::value<string>(&direction_file)->required(),
       "path to the direction file")
      ("start-frame",
       po::value<int>(&start)->default_value(0),
       "start at this second")
      ("total-frames",
       po::value<int>(&frame_count)->default_value(0),
       "number of frames to process")
      ("calib-file",
       po::value<string>(&calib_file)->default_value("data/calib.yml"),
       "path to the calibration file")
      ("mono-calib-file",
       po::value<string>(&mono_calib_file)->default_value(
         "data/calib_phone.yml"),
       "path to the calibration file")
      ("path-trace-file",
       po::value<string>(&path_trace_file),
       "save calculated path to the file")
      ("debug",
       po::bool_switch(&debug)->default_value(false),
       "Start in debug mode.");
 
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  StereoCalibrationData calib(RawStereoCalibrationData::read(calib_file));
  MonoCalibrationData mono_calibration(
    RawMonoCalibrationData::read(mono_calib_file),
    calib.Pl.colRange(0, 3),
    calib.raw.size);

  int frame_width = calib.raw.size.width;
  int frame_height = calib.raw.size.height;

  BagVideoReader rdr(video_file, "/image_raw");
  cv::Mat frame_mat;
  frame_mat.allocator = cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::PAGE_LOCKED);
  frame_mat.create(frame_height, frame_width*2, CV_8UC1);
  cv::Mat mono_frames[] = {
    frame_mat(cv::Range::all(), cv::Range(0, frame_width)),
    frame_mat(cv::Range::all(), cv::Range(frame_width, frame_width*2))
  };

  rdr.skip(start);
    
  int global_frame_index = start-1;

  FrameProcessor frame_processor(calib);

  FrameData frame_data[2] = {
    FrameData(kMaxKeypoints),
    FrameData(kMaxKeypoints)
  };


  CrossFrameProcessorConfig cross_processor_config;
  CrossFrameProcessor cross_processor(calib, cross_processor_config);

  /* DirectionTrackerSettings settings; */
  /* DirectionTarget target = DirectionTarget::read(direction_file); */
  /* DirectionTracker direction_tracker(settings, &calib.intrinsics); */
  /* direction_tracker.setTarget(&target); */

  double frameDt = 1.0/60.0;

  int frame_index = -1;
  int threshold = 25;

  std::vector<cv::Point2d> img_keypoints;

  e::Quaterniond cam_r(1, 0, 0, 0);
  e::Vector3d cam_t(0, 0, 0);

  FILE* trace_file = nullptr;
  if (!path_trace_file.empty()) {
    trace_file = fopen(path_trace_file.c_str(), "w");
    if (trace_file == nullptr) {
      abort();
    }
  }

  bool done = false;
  while (!done && (frame_count == 0 || frame_index + 1 < frame_count)) {
    Timer timer;

    if (!rdr.nextFrame(frame_mat)) {
      break;
    }

    frame_index++;
    global_frame_index++;
    
    std::cout << "Frame #" << global_frame_index << std::endl;

    timer.mark("read");

    FrameData& cur_frame = frame_data[frame_index % 2];
    FrameData& prev_frame = frame_data[1 - (frame_index % 2)];

    frame_processor.process(mono_frames, threshold, cur_frame);

    timer.mark("process");
  
    /* img_keypoints.resize(0); */
    /* for (const auto& kp : processor.keypoints(0)) { */
    /*   img_keypoints.push_back(cv::Point2d(kp.x, kp.y)); */
    /* } */

    /* bool dir_found = */
    /*   direction_tracker.process(img_keypoints, processor.descriptors(0)); */

    /* if (dir_found) { */
    /*   std::cout */ 
    /*     << "Rypr = " */ 
    /*     << rotToEuler(direction_tracker.rot()) * 180 / M_PI << std::endl; */
    /* } */

    timer.mark("dir");

    if (frame_index > 0) {
      e::Quaterniond d_r;
      e::Vector3d d_t;
      e::Matrix3d t_cov;
      bool ok = cross_processor.process(
          prev_frame, cur_frame, d_r, d_t, &t_cov);

      timer.mark("cross");

      if (ok) {
        if (!(t_cov(0, 0) < 5 && t_cov(1, 1) < 5 && t_cov(2, 2) < 5)) {
          std::cout << "FAIL";
          std::cout << "T_cov = " << t_cov << std::endl; 
        } else {
          cam_t += cam_r*d_t;
          cam_r = cam_r*d_r;
          cam_r.normalize();
        }
        
        auto ypr = rotToYawPitchRoll(cam_r) * 180.0 / M_PI;
        std::cout << "yaw = " << ypr.x() 
                  << ", pitch = " << ypr.y() 
                  << ", roll = " << ypr.z() << endl;
        std::cout << "T = " << cam_t.transpose() << endl; 
        /* std::cout << "T_cov = " << cross_processor.t_cov() << std::endl; */

        if (trace_file != nullptr) {
          fprintf(trace_file, "%d 1 %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
              global_frame_index,
              cam_t.x(), cam_t.y(), cam_t.z(), 
              cam_r.w(), cam_r.x(), cam_r.y(), cam_r.z());
        }
      } else {
        if (trace_file) {
          fprintf(trace_file, "%d 0\n", global_frame_index);
        }
        std::cout << "FAIL" << std::endl;
      }
    }

    std::cout << "Times: " << timer.str() << std::endl;

    if (frame_index > 0 && debug) {
      /* DebugRenderer renderer( */
      /*     p1, */ 
      /*     &processor, */ 
      /*     &cross_processor, */ 
      /*     &direction_tracker, */
      /*     &mono_calibration, */
      /*     1600, 1200); */
      /* if (!renderer.loop()) { */
      /*   break; */
      /* } */
    }
  }

  if (trace_file != nullptr) {
    fclose(trace_file);
  }

  return 0;
}
