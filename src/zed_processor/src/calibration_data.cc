#include <opencv2/opencv.hpp>
#include <string>

#include "calibration_data.hpp"

RawStereoCalibrationData RawStereoCalibrationData::read(
    const std::string& filename) {
  RawStereoCalibrationData res;
  
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  fs["size"] >> res.size;
  fs["Ml"] >> res.left_camera.m;
  fs["dl"] >> res.left_camera.d;
  fs["Mr"] >> res.right_camera.m;
  fs["dr"] >> res.right_camera.d;

  cv::Mat om;
  fs["om"] >> om;
  cv::Rodrigues(om, res.R);

  fs["T"] >> res.T;

  return res;
}

void RawStereoCalibrationData::write(const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  cv::write(fs, "size", size);
  cv::write(fs, "Ml", left_camera.m);
  cv::write(fs, "dl", left_camera.d);
  cv::write(fs, "Mr", right_camera.m);
  cv::write(fs, "dr", right_camera.d);

  cv::Mat om;
  cv::Rodrigues(R, om);
  cv::write(fs, "om", om);

  cv::write(fs, "T", T);
}

StereoCalibrationData::StereoCalibrationData(
    const RawStereoCalibrationData& raw) {
  this->raw = raw;

  cv::stereoRectify(
      raw.left_camera.m, raw.left_camera.d, 
      raw.right_camera.m, raw.right_camera.d, 
      raw.size,
      raw.R, raw.T,
      Rl, Rr,
      Pl, Pr,
      Q,
      cv::CALIB_ZERO_DISPARITY,
      0); 

  cv::initUndistortRectifyMap(
      raw.left_camera.m, raw.left_camera.d, Rl, Pl, 
      raw.size,
      CV_32FC1,
      undistort_maps[0].x, undistort_maps[0].y);

  cv::initUndistortRectifyMap(
      raw.right_camera.m, raw.right_camera.d, Rr, Pr, 
      raw.size, 
      CV_32FC1,
      undistort_maps[1].x, undistort_maps[1].y);

  assert(Pl.rows == 3 && Pl.cols == 4);
  assert(Pr.rows == 3 && Pr.cols == 4);

  const cv::Mat_<double>& P1 = static_cast<const cv::Mat_<double>&>(Pl);
  const cv::Mat_<double>& P2 = static_cast<const cv::Mat_<double>&>(Pr);

  intrinsics.f = P1(0, 0);
  assert(P1(1, 1) == intrinsics.f && 
      P2(0, 0) == intrinsics.f && 
      P2(1, 1) == intrinsics.f);
  intrinsics.dr = P2(0, 3); 
  intrinsics.cxl = P1(0, 2);
  intrinsics.cxr = P2(0, 2);
  intrinsics.cy = P1(1, 2);
  assert(P2(1, 2) == intrinsics.cy);
}
