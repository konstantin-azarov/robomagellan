#include <opencv2/opencv.hpp>
#include <vector>

#include "rigid_estimator.hpp"
#include "util.hpp"

using namespace std;

RigidEstimator::RigidEstimator() {
  rot_.create(3, 3, CV_64F);
}

void RigidEstimator::estimate(
    const std::vector<cv::Point3d>& p1,
    const std::vector<cv::Point3d>& p2) {
  pts1_.resize(p1.size());
  std::copy(p1.begin(), p1.end(), pts1_.begin());
  pts2_.resize(p2.size());
  std::copy(p2.begin(), p2.end(), pts2_.begin());

  auto c1 = center(pts1_);
  auto c2 = center(pts2_);

  cv::Mat m1 = cv::Mat(pts1_).reshape(1);
  cv::Mat m2 = cv::Mat(pts2_).reshape(1);

  cv::Mat s = m1.t()*m2;

  cv::SVD svd(s);

  rot_ = (svd.u * svd.vt).t();

  cv::Mat_<double>& r = static_cast<cv::Mat_<double>&>(rot_);
  t_ = c2 - r*c1;
}

cv::Point3d RigidEstimator::center(vector<cv::Point3d>& pts) {
  cv::Point3d center;
  for (int i = 0; i < pts.size(); ++i) {
    center += pts[i];
  }

  center *= 1.0/pts.size();

  for (int i = 0; i < pts.size(); ++i) {
    pts[i] -= center;
  }

  return center;
}
