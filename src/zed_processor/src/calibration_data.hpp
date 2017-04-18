#ifndef __CALIBRATION_DATA__HPP__
#define __CALIBRATION_DATA__HPP__

#include <opencv2/opencv.hpp>
#include <string>

struct CameraCalibrationData {
  cv::Mat m, d;
};

struct RawMonoCalibrationData {
  static RawMonoCalibrationData read(const std::string& filename);
  void write(const std::string& filename);

  cv::Size size;
  CameraCalibrationData camera;
};

struct RawStereoCalibrationData {
  static RawStereoCalibrationData read(const std::string& filename);
  static RawStereoCalibrationData readKitti(
      const std::string& filename, cv::Size img_size);
  void write(const std::string& filename);

  RawStereoCalibrationData resize(double scale);

  cv::Size size;
  CameraCalibrationData left_camera, right_camera;
  cv::Mat R, T;
};

struct UndistortMaps {
  cv::Mat x, y;
};

struct MonoIntrinsics {
  double f, cx, cy;
};

struct MonoCalibrationData {
  MonoCalibrationData() = delete;
  MonoCalibrationData(
      const RawMonoCalibrationData& raw,
      cv::Mat new_m,
      cv::Size target_size);

  RawMonoCalibrationData raw;

  UndistortMaps undistort_maps;
};

struct StereoIntrinsics : public MonoIntrinsics {
  double dr;
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
