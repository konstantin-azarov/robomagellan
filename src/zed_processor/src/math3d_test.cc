#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

#include "math3d.hpp"

namespace e = Eigen;

const double EPS = 1E-9;

TEST(IntersectLines, threeLinesAtOrigin) {
  cv::Mat_<double> l(3, 3);

  l.row(0) = cv::Mat(cv::Vec3d(1, 0, 0)).t();
  l.row(1) = cv::Mat(cv::Vec3d(0, 1, 0)).t();
  l.row(2) = cv::Mat(cv::Vec3d(1, 1, 0)).t();

  auto res = cv::Mat(intersectLines(l));
  ASSERT_TRUE(compareMats(res, cv::Mat(cv::Vec2d(0, 0)), EPS))
      << res;
}


TEST(IntersectLines, threeLinesNotAtOrigin) {
  cv::Mat_<double> l(3, 3);

  l.row(0) = cv::Mat(cv::Vec3d(1, 0, 1)).t();
  l.row(1) = cv::Mat(cv::Vec3d(0, 1, 1)).t();
  l.row(2) = cv::Mat(cv::Vec3d(1, -1, 0)).t();

  auto res = cv::Mat(intersectLines(l));
  ASSERT_TRUE(compareMats(res, cv::Mat(cv::Vec2d(-1, -1)), EPS))
      << res;
}

TEST(RotToEuler, yaw) {
  double a = 30*M_PI/180.0;

  ASSERT_EQ(
      rotToYawPitchRoll(
        e::Quaterniond(e::AngleAxisd(a, e::Vector3d::UnitY()))), 
      e::Vector3d(a, 0, 0));
  ASSERT_EQ(
      rotToYawPitchRoll(
        e::Quaterniond(e::AngleAxisd(-a, e::Vector3d::UnitY()))), 
      e::Vector3d(-a, 0, 0));
}

TEST(RotToEuler, pitch) {
  double a = 30*M_PI/180.0;

  ASSERT_EQ(
      rotToYawPitchRoll(
        e::Quaterniond(e::AngleAxisd(a, e::Vector3d::UnitX()))), 
      e::Vector3d(0, a, 0));
  ASSERT_EQ(
      rotToYawPitchRoll(
        e::Quaterniond(e::AngleAxisd(-a, e::Vector3d::UnitX()))), 
      e::Vector3d(0, -a, 0));
}

TEST(RotToEuler, roll) {
  double a = 30*M_PI/180.0;

  ASSERT_EQ(
      rotToYawPitchRoll(
        e::Quaterniond(e::AngleAxisd(a, e::Vector3d::UnitZ()))), 
      e::Vector3d(0, 0, a));
  ASSERT_EQ(
      rotToYawPitchRoll(
        e::Quaterniond(e::AngleAxisd(-a, e::Vector3d::UnitZ()))), 
      e::Vector3d(0, 0, -a));
}

TEST(RotToEuler, combination) {
  double a = 30*M_PI/180.0;

  auto res = rotToYawPitchRoll(
        e::AngleAxisd(a, e::Vector3d::UnitY())*
        e::AngleAxisd(a/2, e::Vector3d::UnitX())*
        e::AngleAxisd(a/3, e::Vector3d::UnitZ()));

  ASSERT_LT((res - e::Vector3d(a, a/2, a/3)).norm(), 1E-6);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

