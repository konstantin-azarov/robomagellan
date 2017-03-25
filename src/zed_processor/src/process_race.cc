#include <assert.h>
#include <chrono>

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "bag_video_reader.hpp"
#include "calibration_data.hpp"

namespace c = std::chrono;
namespace po = boost::program_options;
namespace e = Eigen;

backward::SignalHandling sh;

const double kWorkingScale =  0.5; 
const int kSegmentLengthThreshold = 3;
const int kMaxSegmentsPerLine = 5;
const float kMaxDistance = 5E+3;
const float kMinDistance = 500;

const int kRansacIterations = 100;
const double kRansacInlierDist = 20;
const int kRansacMinInliers = 20;

const double kGroundMaxAngle = 50 * M_PI / 180;
const double kGroundMinDist = 180;
const double kGroundMaxDist = 270;

const int kMapWidth = 500;
const int kMapHeight = 500;
const double kMapScale = kMapWidth/10E+3;

const double kLaneMaxStartDistance = 100;
const double kLaneMaxContinuationDistance = 1000;
const double kLaneFoldInDistance = 50;
const double kLaneMinContinuationCosine = cos(20 * M_PI / 180.0); 
const int kLaneMinPoints = 15;

const double kInitialWidthEstimate = 2E+3;

const double kMaxSpeed = 5 / 3.6;
const int kSpeedSteps = 4;
const double kMaxTurnRate = 30 * M_PI / 180;
const double kTurnRateSteps = 10;

const double kTrackDistance = 5E+3;

const int kTrackPoints = 20;
const double kTrackTimeStep = kTrackDistance / kMaxSpeed / (kTrackPoints-1);
const double kTrackTailLength = 10E+3;

const double kBestClearTime = 5.0;

StereoIntrinsics scaleIntrinsics(StereoCalibrationData calib, double scale) {
  StereoIntrinsics res;

  res.f = calib.intrinsics.f * scale;
  res.cx = calib.intrinsics.cx * scale;
  res.cy = calib.intrinsics.cy * scale;
  res.dr = calib.intrinsics.dr * scale;

  return res;
}

void extractSegments(
    cv::Mat_<uint8_t> img, int row, std::vector<float>& segments) {
  segments.clear();

  int w = img.cols;

  int start = -1;
  for (int i = 0; i < w; ++i) {
    int p = img(row, i);
    if (p && start == -1) {
      start = i;
    } else if (!p && start > -1 && i - start >= kSegmentLengthThreshold) {
      segments.push_back((i + start - 1.0f)/2.0f);
      start = -1;
    }
  }
}

void buildCandidatePoints(
    StereoIntrinsics calib,
    cv::Mat_<uint8_t> image, 
    std::vector<e::Vector3d>& points) {
  int w = image.cols/2;
  int h = image.rows;

  cv::Mat left = image.colRange(0, w);
  cv::Mat right = image.colRange(w, 2*w);

  std::vector<float> seg_left, seg_right;

  points.clear();

  for (int v = 0; v < h; ++v) {
    extractSegments(left, v, seg_left);
    extractSegments(right, v, seg_right);

    if (seg_left.size() == seg_right.size()) {
      for (int i = 0; i < seg_left.size(); ++i) {
        auto l = seg_left[i];
        auto r = seg_right[i];
        if (l > r) {
          double z = calib.dr / (r - l);

          if (z < kMaxDistance && z > kMinDistance) {
            double x = (l - calib.cx) * z / calib.f;
            double y = (v - calib.cy) * z / calib.f;

            points.push_back(e::Vector3d(x, y, z));
          }
        }
      }
    }
  }
}

bool extractGroundPlane(const std::vector<e::Vector3d>& points, e::Vector4d& plane) {
  if (points.size() < kRansacMinInliers) {
    return false;
  }

  std::uniform_int_distribution<> dist(0, points.size() - 1);
  std::default_random_engine e;

  int best_inlier_count = 0;
  e::Vector3d best_n;
  double best_d = 0;

  for (int t = 0; t < kRansacIterations; ++t) {
    int i1 = dist(e);
    int i2 = dist(e);
    int i3 = dist(e);

    if (i1 == i2 || i1 == i3 || i2 == i3) {
      continue;
    }

    auto r1 = points[i2] - points[i1];
    auto r2 = points[i3] - points[i1];
    auto n = r1.cross(r2);

    if (n.y() > 0) {
      n = -n;
    }

    n.normalize();

    double angle = acos(-n.y());
    if (angle > kGroundMaxAngle) {
      continue;
    }

    double d = -n.dot(points[i2]);

    if (d < kGroundMinDist || d > kGroundMaxDist) {
      continue;
    }

    int inlier_count = 0;
    for (const auto& p : points) {
      double dist = fabs(n.dot(p) + d);

      if (dist < kRansacInlierDist) {
        inlier_count++;
      }
    }

    if (inlier_count > best_inlier_count) {
      best_inlier_count = inlier_count;
      best_n = n;
      best_d = d;
    }
  }

  if (best_inlier_count >= kRansacMinInliers) {
    plane = e::Vector4d(best_n.x(), best_n.y(), best_n.z(), best_d);
    return true;
  }

  return false;
}

void projectPoints(const std::vector<e::Vector3d>& candidate_points,
                  const e::Vector4d& plane,
                  std::vector<e::Vector2d>& projected_points) {
  projected_points.clear();

  e::Vector3d n(plane.x(), plane.y(), plane.z());
  double d = plane.w();

  e::Vector2d origin(-n.x() * d, -n.z() * d);

  for (const auto& p : candidate_points) {
    double t = p.dot(n) + d;
    if (fabs(t) < kRansacInlierDist) {
      auto x = p - n * t;
      projected_points.push_back(e::Vector2d(x.x(), x.z()) - origin);
    }
  }

  std::sort(
      projected_points.begin(), projected_points.end(), 
      [](const e::Vector2d& a, const e::Vector2d& b) {
        return a.y() < b.y();
      });
}

int sign(double v) {
  return v > 0 ? 1 : (v < 0 ? -1 : 0);
}

void extractLanes(const std::vector<e::Vector2d>& points,
                 std::vector<e::Vector2d>& left,
                 std::vector<e::Vector2d>& right) {
  std::vector<std::vector<e::Vector2d>> lanes;

  std::vector<e::Vector2d> potential_lanes;

  for (const auto& p : points) {
    std::vector<e::Vector2d>* best_lane = nullptr;
    double best_angle = 0;
    bool fold_in = false;

    for (auto& l : lanes) {
      e::Vector2d e = l.back();
      e::Vector2d d = l.back() - l[l.size() - 2];
      e::Vector2d d1 = p - e;
      double angle = d1.dot(d) / (d1.norm() * d.norm());

      if (d1.norm() < kLaneFoldInDistance) {
        fold_in = true;
        break;
      }

      if (d1.norm() < kLaneMaxContinuationDistance && angle > best_angle) {
        best_angle = angle;
        best_lane = &l;
      }
    }

    if (fold_in) {
      continue;
    }

    if (best_angle > kLaneMinContinuationCosine) {
      best_lane->push_back(p);
    } else {
      int best_start_index = -1;
      double best_start_distance = kLaneMaxStartDistance;
     
      for (int i = 0; i < potential_lanes.size(); ++i) {
        double d = (potential_lanes[i] - p).norm();
        if (d < best_start_distance) {
          best_start_distance = d;
          best_start_index = i;
        }
      }

      if (best_start_index > -1) {
        std::vector<e::Vector2d> lane = { potential_lanes[best_start_index], p };
        lanes.push_back(lane);
        potential_lanes[best_start_index] = potential_lanes.back();
        potential_lanes.pop_back();
      } else {
        potential_lanes.push_back(p);
      }
    }
  }

  left.clear();
  right.clear();

  int best_left = -1, best_right = -1;
  double best_left_x = -100E+3, best_right_x = 100E+3;

  for (int i=0; i < lanes.size(); ++i) {
    const auto& lane = lanes[i];
    if (lane.size() > kLaneMinPoints) {
      double s = lane[0].x() * lane.back().y() - lane[0].y() * lane.back().x();

      if (s < 0 && lane[0].x() > best_left_x) {
        best_left_x = lane[0].x();
        best_left = i;
      } else if (s > 0 && lane[0].x() < best_right_x) {
        best_right_x = lane[0].x();
        best_right = i;
      }
    }
  }

  if (best_left != -1) {
    left = lanes[best_left];
  }
  if (best_right != -1) {
    right = lanes[best_right];
  }
}

double estimateTrackWidth(
    std::vector<e::Vector2d>& left,
    std::vector<e::Vector2d>& right,
    double old_estimate) {
  int j = 0;
  double sum = 0;
  int cnt = 0;
  for (const auto& l : left) {
    while (j < right.size() && right[j].y() < l.y()) 
      j++;

    if (j == 0) {
      continue;
    }

    if (j == right.size()) {
      break;
    }

    sum += right[j].x() - l.x();
    cnt += 1;
  }

  if (cnt > 5) {
    return sum/cnt;
  } else {
    return old_estimate;
  }
}

double norm_angle(double t) {
  if (t > 2*M_PI) t -= 2*M_PI;
  else if (t < 0) t += 2*M_PI;

  return t;
}

double intersectArcRay(
    double r, const e::Vector2d& p0, const e::Vector2d d,
    double w,
    double& t_arc, double& t_ray) {
  double a = -d.y();
  double b = d.x();
  double c = -(a*p0.x() + b*p0.y());

  double ab = sqrt(a*a + b*b);

  double alpha = atan2(b/ab, a/ab);
  
  double cos_q = (a*r + c)/(r*ab);

  if (fabs(cos_q) > 1) {
    return false;
  }

  double q = acos(cos_q);

  double wt = std::min(norm_angle(alpha + q), norm_angle(alpha - q));

  e::Vector2d p(r*(1 - cos(wt)), r*sin(wt));

  t_ray = d.dot(p - p0)/ab;
  if (t_ray < 0) {
    return false;
  }

  t_arc = wt/w;

  return true;
}

bool intersectSegments(
    const e::Vector2d& p1, const e::Vector2d& p2,
    const e::Vector2d& q1, const e::Vector2d& q2,
    double& t1, double& t2) {
  double x1 = p1.x(), y1 = p1.y();
  double x2 = p2.x(), y2 = p2.y();
  double x3 = q1.x(), y3 = q1.y();
  double x4 = q2.x(), y4 = q2.y();

  double d = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1);

  if (fabs(d) < 1E-6) {
    return false;
  }

  t1 = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / d;
  t2 = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / d;

  if (t1 < 0 || t1 > 1 || t2 < 0 || t2 > 1) {
    return false;
  }

  return true;
}

struct Command {
  double w, v;
  double clear_time;  
};

void discretizeTrack(
    double v, double w, double dt, int n, double tail_length,
    e::Vector2d* res) {
  double r = v/w;

  res[0] = e::Vector2d(0, 0);

  for (int i=1; i < n; ++i) {
    double t = dt*i;
    res[i] = e::Vector2d(r*(1 - cos(w*t)), r*sin(w*t));
  }
  res[n] = res[n-1] + e::Vector2d(sin(w*(n-1)*dt), cos(w*(n-1)*dt))*tail_length;
}

Command evalCommand(
    std::vector<e::Vector2d>& left,
    std::vector<e::Vector2d>& right,
    double v, double w) {
  Command res;


  res.w = w;
  res.v = v;

  e::Vector2d track[kTrackPoints + 1];
  discretizeTrack(v, w, kTrackTimeStep, kTrackPoints, kTrackTailLength, track);

  double t_cross = 1E+10;

  for (int i=1; i < left.size() || i < right.size(); ++i) {
    for (int j=1; j <= kTrackPoints; ++j) {
      double t1, t2;

      if (i < left.size() && 
          intersectSegments(left[i-1], left[i], track[j-1], track[j], t1, t2)) {
        t_cross = std::min(t_cross, t2 + (j-1)*kTrackTimeStep);
        break;
      }

      if (i < right.size() &&
          intersectSegments(right[i-1], right[i], track[j-1], track[j], t1, t2)) {
        t_cross = std::min(t_cross, t2 + (j-1)*kTrackTimeStep);
        break;
      }
    }
  }

  res.clear_time = t_cross;

  return res;
}

void updateSteeringCommand(
    std::vector<e::Vector2d>& left,
    std::vector<e::Vector2d>& right,
    double& v_cmd, double& w_cmd) {
  Command best_command;
  best_command.clear_time = 0;

  for (int k_v = 1; k_v <= kSpeedSteps; ++k_v) {
    for (int k_w = -kTurnRateSteps; k_w <= kTurnRateSteps; k_w++) {
      auto cmd = evalCommand(
          left, right, 
          kMaxSpeed * k_v / kSpeedSteps, 
          kMaxTurnRate * k_w / kTurnRateSteps);

      std::cout << "CMD: v=" << cmd.v << ", w=" << cmd.w << ": " << cmd.clear_time << std::endl; 

      if (cmd.clear_time > best_command.clear_time ||
          (cmd.clear_time > kBestClearTime && best_command.clear_time >= kBestClearTime &&
           cmd.v > best_command.v)) {
        best_command = cmd;
      }
    }
  }

  if (best_command.clear_time > 0) {
    v_cmd = best_command.v;
    w_cmd = best_command.w;
  }
}

bool enhanceLanes(
    std::vector<e::Vector2d>& left,
    std::vector<e::Vector2d>& right,
    double& width_estimate) {
  if (left.empty() && right.empty()) {
    return false;
  }

  if (left.empty()) {
    left.resize(right.size());
    for (int i=0; i < right.size(); ++i) {
      left[i] = e::Vector2d(right[i].x() - width_estimate, right[i].y());
    }
  } else if (right.empty()) {
    right.resize(left.size());
    for (int i=0; i < left.size(); ++i) {
      right[i] = e::Vector2d(left[i].x() + width_estimate, left[i].y());
    }
  } else {
    width_estimate = estimateTrackWidth(left, right, width_estimate);
  }

  return true;
}

void testIntersect() {
  e::Vector2d p1(1, 9), p2(10, -5);
  e::Vector2d q1(2, 2), q2(15, 7);

  double t1, t2;
  cv::Mat dbg_image(200, 200, CV_8UC3);

  cv::line(dbg_image, cv::Point(p1.x()*10, p1.y()*10), cv::Point(p2.x()*10, p2.y()*10), cv::Scalar(0, 0, 255));
  cv::line(dbg_image, cv::Point(q1.x()*10, q1.y()*10), cv::Point(q2.x()*10, q2.y()*10), cv::Scalar(0, 255, 0));

  std::cout << intersectSegments(p1, p2, q1, q2, t1, t2) << std::endl;

  auto r1 = p1 + (p2 - p1)*t1;
  auto r2 = q1 + (q2 - q1)*t2;

  cv::circle(dbg_image, cv::Point(r1.x()*10, r1.y()*10), 3, cv::Scalar(0, 0, 255));
  cv::circle(dbg_image, cv::Point(r2.x()*10, r2.y()*10), 5, cv::Scalar(0, 0, 255));



  std::cout << t1 << " " << t2 << std::endl;

  cv::imshow("debug", dbg_image);
  cv::waitKey(0);
}

int main(int argc, char** argv) {
  std::string video_file, calib_file;

  po::options_description desc("Command line options");
  desc.add_options()
      ("video-file",
       po::value<std::string>(&video_file),
       "path to the video file")
      ("calib-file",
       po::value<std::string>(&calib_file),
       "path to the calibration file");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  auto raw_calib = RawStereoCalibrationData::read(calib_file);
  StereoCalibrationData calib(raw_calib);

  StereoIntrinsics scaled_intrinsics = scaleIntrinsics(calib, kWorkingScale);

  BagVideoReader rdr(video_file, "/image_raw");

//  rdr.skip(100);

  int frame_count = 0;
  int frame_index = -1; 
  bool done = false;

  int frame_width = calib.raw.size.width;
  int frame_height = calib.raw.size.height;

  cv::Mat raw_frame_mat, rectified_mat;
  raw_frame_mat.create(frame_height, frame_width*2, CV_8UC3);
  rectified_mat.create(frame_height, frame_width*2, CV_8UC3);

  cv::Mat frame_mat, hsv_mat, thresholded_img;
  std::vector<e::Vector3d> candidate_points;
  std::vector<e::Vector2d> projected_points;
  std::vector<e::Vector2d> left_lane, right_lane;
  double width_estimate = kInitialWidthEstimate;

  int img_mode = 0, frame_by_frame_mode = true;
 
  int working_width = frame_width * kWorkingScale;
  int working_height = frame_height * kWorkingScale; 

  cv::namedWindow("debug");
  cv::moveWindow("debug", 0, 0);
  cv::namedWindow("map");
  cv::moveWindow("map", 1000, working_height + 50);

  int failed_frames = 0;
    
  double v_cmd = 0, w_cmd = 0;

  while (!done && (frame_count == 0 || frame_index + 1 < frame_count)) {
    frame_index++;

    if (!rdr.nextFrameRaw(raw_frame_mat)) {
      break;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i=0; i < 2; ++i) {
      int c0 = frame_width*i;
      int c1 = c0 + frame_width;
      cv::remap(
          raw_frame_mat.colRange(c0, c1),
          rectified_mat.colRange(c0, c1),
          calib.undistort_maps[i].x,
          calib.undistort_maps[i].y,
          cv::INTER_LINEAR);
    }

    cv::resize(
        rectified_mat, frame_mat, cv::Size(working_width*2, working_height));

    cv::cvtColor(frame_mat, hsv_mat, cv::COLOR_BGR2HSV);

    cv::inRange(
        hsv_mat,
        cv::Scalar(0, 0, 180),
        cv::Scalar(255, 50, 255),
        thresholded_img);

    buildCandidatePoints(scaled_intrinsics, thresholded_img, candidate_points);

    e::Vector4d plane;
      
    if (!extractGroundPlane(candidate_points, plane)) {
      failed_frames++;
      continue;
    }

    projectPoints(candidate_points, plane, projected_points); 
    extractLanes(projected_points, left_lane, right_lane);

    if (left_lane.empty() && right_lane.empty()) {
      failed_frames++;
      continue;
    }
    
    if (!enhanceLanes(left_lane, right_lane, width_estimate)) {
      failed_frames++;
      continue;
    }

    updateSteeringCommand(left_lane, right_lane, v_cmd, w_cmd);
    
    auto t2 = c::high_resolution_clock::now();

    long t_msec = c::duration_cast<c::milliseconds>(t2 - t1).count();

    /* std::cout << "Plane = (" */ 
    /*   << plane.x() << ", " << plane.y() << ", " << plane.z() << ", " */ 
    /*   << plane.w() << ")" << std::endl; */
    /* std::cout << "Angle = " << acos(-plane.y()) * 180/M_PI << std::endl; */

    bool done_debug = false, print_points = false;

    while (!done_debug) {
      done_debug = true;
     
      cv::Mat debug_image, map_img(kMapHeight, kMapWidth, CV_8UC3);
      
      map_img = cv::Scalar(0, 0, 0);

      switch (img_mode) {
        case 0:
          cv::cvtColor(thresholded_img, debug_image, cv::COLOR_GRAY2BGR);
          break;
        case 1:
          hsv_mat.copyTo(debug_image);
          break;
        case 2:
          frame_mat.copyTo(debug_image);
          break;
      }

      for (auto p : candidate_points) {
        double d = plane.x() * p.x() + plane.y() * p.y() + plane.z() * p.z() + plane.w();
        
        double u = (p.x() * scaled_intrinsics.f) / p.z() + scaled_intrinsics.cx;
        double u1 = (p.x() * scaled_intrinsics.f + scaled_intrinsics.dr) / p.z() + scaled_intrinsics.cx; 
        double v = (p.y() * scaled_intrinsics.f) / p.z() + scaled_intrinsics.cy;

        cv::Scalar color(0, 0, 255);

        if (fabs(d) < kRansacInlierDist) {
          if (print_points) {
            std::cout << p.x() << ", " << p.y() << ", " << p.z() << ": " << u << ", " << v << ": " << d <<  std::endl;
          }

          color = cv::Scalar(255, 0, 0);

        }
        cv::circle(debug_image, cv::Point(u, v), 5, color);
        cv::circle(debug_image, cv::Point(u1 + working_width, v), 5, color);
      }

      float cx = kMapWidth/2;
      float cy = kMapHeight;
      float s = kMapScale;
      for (const auto& p : projected_points) {
        cv::circle(map_img, cv::Point(cx + p.x()*s, cy - p.y()*s), 2, cv::Scalar(255, 255, 255));
      }

      for (const auto* l : {&left_lane, &right_lane}) {
        for (int i = 1; i < l->size(); ++i) {
          auto p1 = (*l)[i-1];
          auto p2 = (*l)[i];
          cv::line(
              map_img, 
              cv::Point(cx + p1.x()*s, cy - p1.y()*s), 
              cv::Point(cx + p2.x()*s, cy - p2.y()*s),
              l == &left_lane ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0),
              1);

        }
      }

      cv::line(map_img, cv::Point(cx - 5, cy), cv::Point(cx + 5, cy), cv::Scalar(0, 0, 255));
      cv::line(map_img, cv::Point(cx, cy - 5), cv::Point(cx, cy + 5), cv::Scalar(0, 0, 255)); 

      // Render trajectory
      e::Vector2d track[kTrackPoints + 1];
      discretizeTrack(
          v_cmd, w_cmd, kTrackTimeStep, kTrackPoints, kTrackTailLength, 
          track);


      for (int i=1; i <= kTrackPoints; ++i) {
        e::Vector2d p = track[i-1], c = track[i];
        cv::line(
            map_img, 
            cv::Point(cx + p.x()*s, cy - p.y()*s), 
            cv::Point(cx + c.x()*s, cy - c.y()*s),
            cv::Scalar(0, 255, 255));
      }
          
      cv::imshow("debug", debug_image);
      cv::imshow("map", map_img);

      int key = cv::waitKey(frame_by_frame_mode ? 0 : 1);
      if (key != -1) key &= 0xFF;
      switch (key) {
        case 27:
          done = true;
          break;
        case 'p':
          print_points = !print_points;
          done_debug = false;
          break;
        case 'c':
          img_mode = (img_mode + 1) % 3;
          done_debug = false;
          break;
        case 'r':
          done_debug = true;
          frame_by_frame_mode = !frame_by_frame_mode;
          break;
      }
    }
  } 

  std::cout << "Total frames: " << frame_index + 1 
    << "; failed: " << failed_frames << std::endl;
}

