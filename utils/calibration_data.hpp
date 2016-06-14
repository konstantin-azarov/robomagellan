#ifndef __CALIBRATION_DATA__HPP__
#define __CALIBRATION_DATA__HPP__

#include <opencv2/opencv.hpp>
#include <string>

struct StereoIntrinsics {
  double f, dr, cxl, cxr, cy;
};

struct UndistortMaps {
  cv::Mat x, y;
};

struct CalibrationData {
  UndistortMaps undistortMaps[2];
  StereoIntrinsics intrinsics;
  cv::Mat Q;

  static CalibrationData read(const std::string& filename, int width, int height);
};



#endif
