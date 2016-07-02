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

struct RawCalibrationData {
  static RawCalibrationData read(
      const std::string& filename);

  cv::Size size;
  cv::Mat Ml, dl, Mr, dr;
  cv::Mat R, T;
};

struct CalibrationData {
  CalibrationData(const RawCalibrationData& raw);

  UndistortMaps undistortMaps[2];
  StereoIntrinsics intrinsics;
  cv::Mat Q;

 private:
  CalibrationData() {}

};


#endif
