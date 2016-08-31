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

struct CameraCalibrationData {
  cv::Mat M, d;
};

struct StereoCalibrationData {
  cv::Mat R, T;
}

struct RawCalibrationData {
  static RawCalibrationData read(
      const std::string& filename);

  void write(const std::string& filename);

  cv::Size size;
  CameraCalibrationData left, right;
  StereoCalibrationData stereo;
};

struct CalibrationData {
  CalibrationData(const RawCalibrationData& raw);

  UndistortMaps undistortMaps[2];
  StereoIntrinsics intrinsics;
  cv::Mat Rl, Rr, Pl, Pr, Q;
  RawCalibrationData raw;

 private:
  CalibrationData() {}

};


#endif
