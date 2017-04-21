#ifndef __CONE_TRACKER__HPP__
#define __CONE_TRACKER__HPP__

#include <Eigen/Core>

#include <opencv2/core.hpp>

#include "calibration_data.hpp"

class ConeTracker {
  public:
    ConeTracker(const StereoCalibrationData& calib);

    void process(const cv::Mat& frame, std::vector<Eigen::Vector3d>& cones); 

  private:
    StereoCalibrationData calib_;
    cv::Mat b1_, b2_, binary_img_, labels_, centroids_;
    cv::Mat_<int32_t> stats_;
    std::vector<cv::Rect2i> left_rects_, right_rects_;
};

#endif
