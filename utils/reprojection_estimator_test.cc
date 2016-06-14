#define CATCH_CONFIG_MAIN

#include <cmath>
#include <opencv2/opencv.hpp>
#include <random>

#include "catch.hpp"

#include "reprojection_estimator.hpp"

#include "math3d.hpp"

const double EPS = 1E-7;

std::vector<std::pair<cv::Point3d, cv::Point3d>> regularPoints(cv::Mat R, cv::Point3d t) {
  std::vector<cv::Point3d> src, transformed;
  src.push_back(cv::Point3d(0, 0, 2));
  src.push_back(cv::Point3d(1, 0, 1));
  src.push_back(cv::Point3d(0, 1, 1));
  src.push_back(cv::Point3d(10, 10, 10));

  std::vector<std::pair<cv::Point3d, cv::Point3d>> res;

  for (auto p : src) {
    res.push_back(std::make_pair(p, R*p + t));
  }

  return res;
}

void testRegularPoints(const StereoIntrinsics& intrinsics, cv::Mat r, cv::Point3d t) {
  ReprojectionEstimator estimator(&intrinsics);
  
  auto pts = regularPoints(r, t);

  std::vector<ReprojectionFeature> features;
  for (auto p : pts) {
    auto p1 = projectPoint(intrinsics, p.first);
    auto p2 = projectPoint(intrinsics, p.second);

    features.push_back(ReprojectionFeature { 
        p.first, p.second, p1.first, p1.second, p2.first, p2.second
    }); 
  }
  
  estimator.estimate(features);

  if (!compareMats(estimator.rot(), r, EPS)) {
    std::cerr << "Expected: " << r << "got: " << estimator.rot() << std::endl;
    REQUIRE(false);
  }

  REQUIRE(comparePoints(estimator.t(), t, EPS));
}


TEST_CASE("Reprojection estimator", "[ReprojectionEstimator]") {
  StereoIntrinsics intrinsics;
  intrinsics.f = 100;
  intrinsics.dr = 500;
  intrinsics.cxl = 320;
  intrinsics.cxr = 420;
  intrinsics.cy = 200;

  SECTION("rotX") {
    testRegularPoints(intrinsics, rotX(10*M_PI/180), cv::Point3d());
  }
  SECTION("rotY") {
    testRegularPoints(intrinsics, rotY(10*M_PI/180), cv::Point3d());
  }
  SECTION("rotZ") {
    testRegularPoints(intrinsics, rotZ(10*M_PI/180), cv::Point3d());
  }
  SECTION("tX") {
    testRegularPoints(intrinsics, cv::Mat::eye(3, 3, CV_64FC1), cv::Point3d(10, 0, 0));
  }
  SECTION("tY") {
    testRegularPoints(intrinsics, cv::Mat::eye(3, 3, CV_64FC1), cv::Point3d(0, 10, 0));
  }
  SECTION("tZ") {
    testRegularPoints(intrinsics, cv::Mat::eye(3, 3, CV_64FC1), cv::Point3d(0, 0, 10));
  }
  SECTION("random") {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> angle_d(-M_PI*30/180, M_PI*30/180);
    std::uniform_real_distribution<double> translation_d(0.0, 10.0);

    for (int i = 0; i < 100; ++i) {
      auto angle = std::bind(angle_d, generator);
      auto t = std::bind(translation_d, generator);

      testRegularPoints(
          intrinsics,
          rotX(angle())*rotY(angle())*rotZ(angle()),
          cv::Point3d(t(), t(), t()));
    }
  }
}
