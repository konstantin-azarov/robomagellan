#include "cone_tracker.hpp"

#include <iostream>

const int kMinWidth = 6;
const double kMinAspectRatio = 1.2;
const int kMaxYDiscrepancy = 3;
const double kMaxConeDistance = 15E+3;

ConeTracker::ConeTracker(const StereoCalibrationData& calib) 
  : calib_(calib) {
}

void ConeTracker::process(
    const cv::Mat& frame,
    std::vector<Eigen::Vector3d>& cones) {
  int img_w = frame.cols/2;

  cv::inRange(
      frame, 
      cv::Scalar(0, 175, 0),
      cv::Scalar(25, 255, 255),
      b1_);

  cv::inRange(
      frame, 
      cv::Scalar(155, 175, 0),
      cv::Scalar(180, 255, 255),
      b2_);

  cv::bitwise_or(b1_, b2_, binary_img_);

  cv::erode(
      binary_img_, binary_img_,
      getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
  cv::dilate(
      binary_img_, binary_img_,
      getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4)));

  cv::connectedComponentsWithStats(
      binary_img_, labels_, stats_, centroids_, 4);

  left_rects_.resize(0);
  right_rects_.resize(0);

  for (int i=0; i < stats_.rows; ++i) {
    int x = stats_(i, cv::CC_STAT_LEFT);
    int y = stats_(i, cv::CC_STAT_TOP);
    int w = stats_(i, cv::CC_STAT_WIDTH);
    int h = stats_(i, cv::CC_STAT_HEIGHT);

    if (w > kMinWidth && h > w * kMinAspectRatio) {
      if (x + w < img_w) {
        left_rects_.push_back(cv::Rect2i(x, y, w, h));
      } else if (x > img_w) {
        right_rects_.push_back(cv::Rect2i(x - img_w, y, w, h));
      }
    }
  }

  const auto& c = calib_.intrinsics;

  cones.clear();

  for (const auto& l : left_rects_) {
    for (const auto& r : right_rects_) {
      if (abs(l.y - r.y) < kMaxYDiscrepancy &&
          abs(l.y + l.height - r.y - r.height) < kMaxYDiscrepancy) {
        double d = -(l.x - r.x + (l.width - r.width)/2); 
        double z = c.dr / d;
        if (z >= 0) {
          if (z < kMaxConeDistance) {
            double x = (l.x + l.width/2 - c.cx) * z / c.f;
            double y = ((l.y + l.height + r.y + r.height) / 2.0 - c.cy) * z / c.f;

            cones.push_back(Eigen::Vector3d(x, y, z));
          }
        }
      }
    }
  }

  /* cv::imshow("threshold", binary_img_); */

  /* if (cv::waitKey(1) == 's') { */
  /*   cv::Mat bgr; */
  /*   cv::cvtColor(frame, bgr, cv::COLOR_HSV2BGR); */

  /*   cv::imwrite("/home/konstantin/cone.png", bgr); */
  /* } */
}
