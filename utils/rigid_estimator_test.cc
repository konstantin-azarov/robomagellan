#define CATCH_CONFIG_MAIN

#include <cmath>
#include <opencv2/opencv.hpp>
#include <random>

#include "catch.hpp"
#include "rigid_estimator.hpp"

const double EPS = 1E-10;

std::vector<std::pair<cv::Point3d, cv::Point3d>> regularPoints(cv::Mat R, cv::Point3d t) {
  std::vector<cv::Point3d> src, transformed;
  src.push_back(cv::Point3d(0, 0, 0));
  src.push_back(cv::Point3d(0, 0, 1));
  src.push_back(cv::Point3d(0, 1, 0));
  src.push_back(cv::Point3d(1, 0, 0));

  cv::Mat_<double>& r = static_cast<cv::Mat_<double>&>(R);
  for (auto p : src) {
    transformed.push_back(cv::Point3d(
      r(0, 0) * p.x + r(0, 1) * p.y + r(0, 2) * p.z + t.x,
      r(1, 0) * p.x + r(1, 1) * p.y + r(1, 2) * p.z + t.y,
      r(2, 0) * p.x + r(2, 1) * p.y + r(2, 2) * p.z + t.z
    ));
  }

  std::vector<std::pair<cv::Point3d, cv::Point3d>> res;

  for (int i = 0; i < src.size(); ++i) {
    res.push_back(std::make_pair(src[i], transformed[i]));
  }

  return res;
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

bool compareMats(const cv::Mat& l, const cv::Mat& r) {
  for (int i=0; i < 3; ++i) {
    for (int j=0; j < 3; ++j) {
      if (std::abs(l.at<double>(i, j) - r.at<double>(i, j)) > EPS) {
        return false;
      }
    }
  }

  return true;
}

bool comparePoints(const cv::Point3d& l, const cv::Point3d& r) {
  if (std::abs(l.x - r.x) > EPS) {
    return false;
  }
  if (std::abs(l.y - r.y) > EPS) {
    return false;
  }
  if (std::abs(l.z - r.z) > EPS) {
    return false;
  }
  return true;
}

void testRegularPoints(cv::Mat r, cv::Point3d t) {
  RigidEstimator estimator;

  estimator.estimate(regularPoints(r, t));

  REQUIRE(compareMats(estimator.rot(), r));
  REQUIRE(comparePoints(estimator.t(), t));
}

TEST_CASE("Rigid estimator", "[RigidEstimator]") {
  SECTION("rotX") {
    testRegularPoints(rotX(30*M_PI/180), cv::Point3d());
  }
  SECTION("rotY") {
    testRegularPoints(rotY(30*M_PI/180), cv::Point3d());
  }
  SECTION("rotZ") {
    testRegularPoints(rotZ(30*M_PI/180), cv::Point3d());
  }
  SECTION("tX") {
    testRegularPoints(cv::Mat::eye(3, 3, CV_64FC1), cv::Point3d(10, 0, 0));
  }
  SECTION("tY") {
    testRegularPoints(cv::Mat::eye(3, 3, CV_64FC1), cv::Point3d(0, 10, 0));
  }
  SECTION("tZ") {
    testRegularPoints(cv::Mat::eye(3, 3, CV_64FC1), cv::Point3d(0, 0, 10));
  }
  SECTION("random") {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> angle_d(-M_PI, M_PI);
    std::uniform_real_distribution<double> translation_d(-10.0, 10.0);

    for (int i = 0; i < 100; ++i) {
      auto angle = std::bind(angle_d, generator);
      auto t = std::bind(translation_d, generator);

      testRegularPoints(
          rotX(angle())*rotY(angle())*rotZ(angle()),
          cv::Point3d(t(), t(), t()));
    }
  }
}
