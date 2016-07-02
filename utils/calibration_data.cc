#include <opencv2/opencv.hpp>
#include <string>

#include "calibration_data.hpp"

RawCalibrationData RawCalibrationData::read(const std::string& filename) {
  RawCalibrationData res;
  
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  fs["size"] >> res.size;
  fs["Ml"] >> res.Ml;
  fs["dl"] >> res.dl;
  fs["Mr"] >> res.Mr;
  fs["dr"] >> res.dr;
  fs["R"] >> res.R;
  fs["T"] >> res.T;

  return res;
}

CalibrationData::CalibrationData(const RawCalibrationData& raw) {
  CalibrationData res;
  cv::Mat Rl, Rr, Pl, Pr;

  cv::stereoRectify(
      raw.Ml, raw.dl, 
      raw.Mr, raw.dr, 
      raw.size,
      raw.R, raw.T,
      Rl, Rr,
      Pl, Pr,
      res.Q,
      cv::CALIB_ZERO_DISPARITY,
      0); 

  cv::initUndistortRectifyMap(
      raw.Ml, raw.dl, Rl, Pl, 
      raw.size,
      CV_32FC1,
      res.undistortMaps[0].x, res.undistortMaps[0].y);

  cv::initUndistortRectifyMap(
      raw.Mr, raw.dr, Rr, Pr, 
      raw.size, 
      CV_32FC1,
      res.undistortMaps[1].x, res.undistortMaps[1].y);

  assert(Pl.rows == 3 && Pl.cols == 4);
  assert(Pr.rows == 3 && Pr.cols == 4);

  const cv::Mat_<double>& P1 = static_cast<const cv::Mat_<double>&>(Pl);
  const cv::Mat_<double>& P2 = static_cast<const cv::Mat_<double>&>(Pr);

  StereoIntrinsics& intrinsics = res.intrinsics;
  
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
