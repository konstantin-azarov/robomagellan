#include <assert.h>
#include <opencv2/opencv.hpp>

#include "math3d.hpp"

cv::Point3d operator * (const cv::Mat& m, const cv::Point3d& p) {
  assert(m.rows == 3 && m.cols == 3);

  const cv::Mat_<double>& r = static_cast<const cv::Mat_<double>&>(m);

  return cv::Point3d(r(0, 0) * p.x + r(0, 1) * p.y + r(0, 2) * p.z,
                     r(1, 0) * p.x + r(1, 1) * p.y + r(1, 2) * p.z,
                     r(2, 0) * p.x + r(2, 1) * p.y + r(2, 2) * p.z);
}

cv::Point2f projectPoint(const cv::Mat& m, const cv::Point3d& p) {
  assert(m.rows == 3 && m.cols == 4);
  
  const cv::Mat_<double>& P = static_cast<const cv::Mat_<double>&>(m);

  return cv::Point2f((P(0, 0) * p.x + P(0, 2) * p.z + P(0, 3))/p.z,
                     (P(1, 1) * p.y + P(1, 2) * p.z)/p.z);
}

std::pair<cv::Point2d, cv::Point2d> projectPoint(
    const StereoIntrinsics& i, 
    const cv::Point3d& p) {

  double v = i.f * p.y / p.z + i.cy;
  cv::Point2d left(i.f * p.x / p.z + i.cxl, v);
  cv::Point2d right((i.f * p.x + i.dr) / p.z + i.cxr, v);

  return std::make_pair(left, right);
}

cv::Mat rotX(double angle) {
  cv::Mat res = cv::Mat::eye(3, 3, CV_64FC1);
  cv::Mat_<double>& r = static_cast<cv::Mat_<double>&>(res);

  double c = cos(angle);
  double s = sin(angle);

  r(1, 1) = c;
  r(1, 2) = -s;
  r(2, 1) = s;
  r(2, 2) = c;

  return res;
}

cv::Mat rotY(double angle) {
  cv::Mat res = cv::Mat::eye(3, 3, CV_64FC1);
  cv::Mat_<double>& r = static_cast<cv::Mat_<double>&>(res);

  double c = cos(angle);
  double s = sin(angle);

  r(0, 0) = c;
  r(0, 2) = s;
  r(2, 0) = -s;
  r(2, 2) = c;

  return res;
}

cv::Mat rotZ(double angle) {
  cv::Mat res = cv::Mat::eye(3, 3, CV_64FC1);
  cv::Mat_<double>& r = static_cast<cv::Mat_<double>&>(res);

  double c = cos(angle);
  double s = sin(angle);

  r(0, 0) = c;
  r(0, 1) = -s;
  r(1, 0) = s;
  r(1, 1) = c;

  return res;
}

bool compareMats(const cv::Mat& l, const cv::Mat& r, double eps) {
  for (int i=0; i < 3; ++i) {
    for (int j=0; j < 3; ++j) {
      if (std::abs(l.at<double>(i, j) - r.at<double>(i, j)) > eps) {
        return false;
      }
    }
  }

  return true;
}

bool comparePoints(const cv::Point3d& l, const cv::Point3d& r, double eps) {
  if (std::abs(l.x - r.x) > eps) {
    return false;
  }
  if (std::abs(l.y - r.y) > eps) {
    return false;
  }
  if (std::abs(l.z - r.z) > eps) {
    return false;
  }
  return true;
}


double norm2(const cv::Point2f& pt) {
  return pt.x*pt.x + pt.y*pt.y;
}
