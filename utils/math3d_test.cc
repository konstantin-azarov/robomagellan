#define CATCH_CONFIG_MAIN

#include <opencv2/opencv.hpp>

#include "catch.hpp"

#include "math3d.hpp"

const double EPS = 1E-9;

TEST_CASE("intersectLines") {
  SECTION("three lines, origin") {
    cv::Mat_<double> l(3, 3);

    l.row(0) = cv::Mat(cv::Vec3d(1, 0, 0));
    l.row(1) = cv::Mat(cv::Vec3d(0, 1, 0));
    l.row(2) = cv::Mat(cv::Vec3d(1, 1, 0));

    REQUIRE(
        compareMats(cv::Mat(intersectLines(l)), cv::Mat(cv::Vec2d(0, 0)), EPS));
  }

  SECTION("three lines, off origin") {
    cv::Mat_<double> l(3, 3);

    l.row(0) = cv::Mat(cv::Vec3d(1, 0, 1));
    l.row(1) = cv::Mat(cv::Vec3d(0, 1, 1));
    l.row(2) = cv::Mat(cv::Vec3d(1, 1, 0));

    REQUIRE(
        compareMats(cv::Mat(intersectLines(l)), cv::Mat(cv::Vec2d(1, 1)), EPS));
  }
}
