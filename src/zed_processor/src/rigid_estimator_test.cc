#define CATCH_CONFIG_MAIN

#include <cmath>
#include <opencv2/opencv.hpp>
#include <random>

#include "catch.hpp"

#include "rigid_estimator.hpp"

#include "math3d.hpp"

const double EPS = 1E-10;

std::pair<std::vector<cv::Point3d>, 
          std::vector<cv::Point3d>> regularPoints(cv::Mat R, cv::Point3d t) {
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

  return std::make_pair(src, transformed);
}

void testRegularPoints(cv::Mat r, cv::Point3d t) {
  RigidEstimator estimator;
  
  auto pts = regularPoints(r, t);
  
  estimator.estimate(pts.first, pts.second);

  if (!compareMats(estimator.rot(), r, EPS)) {
    std::cerr << "Expected: " << r << "got: " << estimator.rot() << std::endl;
    REQUIRE(false);
  }
  REQUIRE(comparePoints(estimator.t(), t, EPS));
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
