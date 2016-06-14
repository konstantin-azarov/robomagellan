#include <opencv2/opencv.hpp>
#include <string>

#include "calibration_data.hpp"

CalibrationData CalibrationData::read(
    const std::string& filename, int width, int height) {
  cv::Mat Ml, dl, Mr, dr, R, T;

  cv::FileStorage fs(filename, cv::FileStorage::READ);

  fs["Ml"] >> Ml;
  fs["dl"] >> dl;
  fs["Mr"] >> Mr;
  fs["dr"] >> dr;
  fs["R"] >> R;
  fs["T"] >> T;

  CalibrationData res;
  cv::Mat Rl, Rr, Pl, Pr;

  cv::stereoRectify(
      Ml, dl, 
      Mr, dr, 
      cv::Size(width, height),
      R, T,
      Rl, Rr,
      Pl, Pr,
      res.Q,
      cv::CALIB_ZERO_DISPARITY,
      0); 

  cv::initUndistortRectifyMap(
      Ml, dl, Rl, Pl, 
      cv::Size(width, height),
      CV_32FC1,
      res.undistortMaps[0].x, res.undistortMaps[0].y);

  cv::initUndistortRectifyMap(
      Mr, dr, Rr, Pr, 
      cv::Size(width, height),
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

  return res;
}
