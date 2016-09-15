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

double norm3(const cv::Point3d& pt) {
  return pt.x*pt.x + pt.y*pt.y + pt.z*pt.z;
}

double descriptorDist(const cv::Mat& a, const cv::Mat& b) {
  assert(a.rows == 1 && b.rows == 1 && a.cols == b.cols);
  assert(a.type() == CV_8UC1 && b.type() == CV_8UC1);

  return cv::norm(a, b, cv::NORM_HAMMING);
//
//  return cv::norm(a, b);
}

cv::Mat hconcat(const cv::Mat& l, const cv::Mat& r) {
  cv::Mat res;
  cv::hconcat(l, r, res);
  return res;
}

cv::Mat leastSquares(const cv::Mat& x, const cv::Mat& y) {
  return (x.t()*x).inv()*x.t()*y;
}

cv::Vec3d fitLine(const cv::Mat& pts) {
  assert(pts.cols == 2 && pts.channels() == 1);

  int n = pts.rows;

  auto x = hconcat(pts.col(0), cv::Mat::ones(n, 1, pts.type()));
  auto y = pts.col(1);

  cv::Mat_<double> params = leastSquares(x, y);

  double a = params(0, 0);
  double b = -1;
  double c = params(0, 1);
  double d = sqrt(a*a + b*b);

  return cv::Vec3d(a/d, b/d, c/d);
}

cv::Vec2d intersectLines(const cv::Mat& lines) {
  assert(lines.cols == 3 && lines.channels() == 1);

  cv::Mat_<double> a(2, 3);

  for (int i=0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      a(i, j) = cv::Mat_<double>(lines.col(i).t() * lines.col(j))(0, 0);
    }
  }

  cv::Mat_<double> r = -a.colRange(0, 2).inv()*a.col(2);

  return cv::Vec2d(r(0, 0), r(1, 0));
}

