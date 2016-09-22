#ifndef __CALIBRATION_DATA__HPP__
#define __CALIBRATION_DATA__HPP__

#include <opencv2/opencv.hpp>
#include <string>

struct CameraCalibrationData {
  cv::Mat m, d;
};

struct RawMonoCalibrationData {
  cv::Size size;
  CameraCalibrationData camera;
};

struct RawStereoCalibrationData {
  static RawStereoCalibrationData read(
      const std::string& filename);

  void write(const std::string& filename);

  cv::Size size;
  CameraCalibrationData left_camera, right_camera;
  cv::Mat R, T;
};

struct UndistortMaps {
  cv::Mat x, y;
};

struct MonoCalibrationData {
  MonoCalibrationData() = delete;
  MonoCalibrationData(const RawMonoCalibrationData& raw);

  CameraCalibrationData raw;

  UndistortMaps undistortMaps;
};

struct StereoIntrinsics {
  double f, dr, cxl, cxr, cy;
};

struct StereoCalibrationData {
  StereoCalibrationData() = delete;
  StereoCalibrationData(const RawStereoCalibrationData& raw);

  RawStereoCalibrationData raw;
  UndistortMaps undistort_maps[2];
  StereoIntrinsics intrinsics;
  cv::Mat Rl, Rr, Pl, Pr, Q;
};


#endif
