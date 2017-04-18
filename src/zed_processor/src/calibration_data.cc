#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

#include "calibration_data.hpp"

RawMonoCalibrationData RawMonoCalibrationData::read(
    const std::string& filename) {
  RawMonoCalibrationData res;

  cv::FileStorage fs(filename, cv::FileStorage::READ);

  fs["size"] >> res.size;
  fs["M"] >> res.camera.m;
  fs["d"] >> res.camera.d;

  return res;
}

void RawMonoCalibrationData::write(const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  cv::write(fs, "size", size);
  cv::write(fs, "M", camera.m);
  cv::write(fs, "d", camera.d);
}

MonoCalibrationData::MonoCalibrationData(
    const RawMonoCalibrationData& raw_p,
    cv::Mat new_m,
    cv::Size target_size) : raw(raw_p) {
  cv::initUndistortRectifyMap(
      raw.camera.m,
      raw.camera.d,
      cv::noArray(),
      new_m,
      target_size,
      CV_32FC1,
      undistort_maps.x,
      undistort_maps.y);
}

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

RawStereoCalibrationData RawStereoCalibrationData::readKitti(
    const std::string& filename,
    cv::Size img_size) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cerr << "File not found: " << filename << std::endl;
    abort();
  }

  auto read_camera_mat = [&f]() {
    std::string name;
    f >> name;

    cv::Mat_<double> m(3, 4);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j ) {
        f >> m(i, j);
      }
    }

    return m;
  };

  RawStereoCalibrationData res;
  res.size = img_size;

  auto m = read_camera_mat();
  res.left_camera.m = m.colRange(0, 3);
  if (cv::countNonZero(m.col(3))) {
    abort();
  }
  res.left_camera.d = cv::Mat_<double>::zeros(1, 5);

  m = read_camera_mat();
  res.right_camera.m = m.colRange(0, 3);
  if (cv::countNonZero(m.col(3).rowRange(1, 3))) {
    abort();
  }
  res.right_camera.d = cv::Mat_<double>::zeros(1, 5);

  res.R = cv::Mat_<double>::eye(3, 3);

  cv::Mat_<double> t = cv::Mat_<double>::zeros(3, 1);
  t(0, 0) = 1000 * (m(0, 3)/m(0, 0));
  res.T = t;

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

RawStereoCalibrationData RawStereoCalibrationData::resize(double scale) {
  RawStereoCalibrationData res;

  res.size = cv::Size(scale * size.width, scale * size.height);
  res.left_camera.m = left_camera.m * scale;
  res.left_camera.d = left_camera.d;
  res.right_camera.m = right_camera.m * scale;
  res.right_camera.d = right_camera.d;
  res.R = R;
  res.T = T;

  return res;
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

  auto verify = [](bool p) {
    if (!p) {
      abort();
    }
  };

  verify(Pl.rows == 3 && Pl.cols == 4);
  verify(Pr.rows == 3 && Pr.cols == 4);

  const cv::Mat_<double>& P1 = static_cast<const cv::Mat_<double>&>(Pl);
  const cv::Mat_<double>& P2 = static_cast<const cv::Mat_<double>&>(Pr);

  intrinsics.f = P1(0, 0);
  verify(P1(1, 1) == intrinsics.f && 
      P2(0, 0) == intrinsics.f && 
      P2(1, 1) == intrinsics.f);
  intrinsics.dr = P2(0, 3); 
  
  intrinsics.cx = P1(0, 2);
  verify(P2(0, 2) == intrinsics.cx);

  intrinsics.cy = P1(1, 2);
  verify(P2(1, 2) == intrinsics.cy);
}
