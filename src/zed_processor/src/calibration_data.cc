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

  cv::Mat om;
  fs["om"] >> om;
  cv::Rodrigues(om, res.R);

  fs["T"] >> res.T;

  return res;
}

void RawCalibrationData::write(const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  cv::write(fs, "size", size);
  cv::write(fs, "Ml", Ml);
  cv::write(fs, "dl", dl);
  cv::write(fs, "Mr", Mr);
  cv::write(fs, "dr", dr);

  cv::Mat om;
  cv::Rodrigues(R, om);
  cv::write(fs, "om", om);

  cv::write(fs, "T", T);
}

CalibrationData::CalibrationData(const RawCalibrationData& raw) {
  this->raw = raw;

  cv::stereoRectify(
      raw.Ml, raw.dl, 
      raw.Mr, raw.dr, 
      raw.size,
      raw.R, raw.T,
      Rl, Rr,
      Pl, Pr,
      Q,
      cv::CALIB_ZERO_DISPARITY,
      0); 

  cv::initUndistortRectifyMap(
      raw.Ml, raw.dl, Rl, Pl, 
      raw.size,
      CV_32FC1,
      undistortMaps[0].x, undistortMaps[0].y);

  cv::initUndistortRectifyMap(
      raw.Mr, raw.dr, Rr, Pr, 
      raw.size, 
      CV_32FC1,
      undistortMaps[1].x, undistortMaps[1].y);

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
