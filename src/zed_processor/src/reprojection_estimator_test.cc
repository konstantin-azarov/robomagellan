#define CATCH_CONFIG_MAIN

#include <cmath>
#include <opencv2/opencv.hpp>
#include <random>

#include <gtest/gtest.h>

#include "reprojection_estimator.hpp"

#include "math3d.hpp"

const double EPS = 1E-6;

class ReprojectionEstimatorTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      intrinsics.f = 100;
      intrinsics.dr = 500;
      intrinsics.cx = 320;
      intrinsics.cy = 200;
    }

    std::vector<std::pair<cv::Point3d, cv::Point3d>> regularPoints(
        cv::Mat R, cv::Point3d t) {
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

    std::vector<StereoReprojectionFeature> stereoFeatures(
        const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts) {
      std::vector<StereoReprojectionFeature> features;
      for (auto p : pts) {
        auto p1 = projectPoint(intrinsics, p.first);
        auto p2 = projectPoint(intrinsics, p.second);

        features.push_back(StereoReprojectionFeature { 
            p.first, p.second, p1.first, p1.second, p2.first, p2.second
        }); 
      }

      return features;
    }

    cv::Point2d projectMono(const cv::Point3d& pt) {
      return cv::Point2d(
        pt.x/pt.z * intrinsics.f + intrinsics.cx,
        pt.y/pt.z * intrinsics.f + intrinsics.cy);
    }

    std::vector<MonoReprojectionFeature> monoFeatures(
        const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts) {
      std::vector<MonoReprojectionFeature> features;
      for (auto p : pts) {
        auto s1 = projectMono(p.first);
        auto s2 = projectMono(p.second);

        features.push_back(MonoReprojectionFeature { 
            p.first, p.second, s1, s2
        });
      }

      return features;
    }

    void testRegularPointsStereo(cv::Mat r, cv::Point3d t) {
      StereoReprojectionEstimator estimator(&intrinsics);
      
      auto pts = regularPoints(r, t);
      auto features = stereoFeatures(pts);
      
      estimator.estimate(features);

      ASSERT_TRUE(compareMats(estimator.rot(), r, EPS))
        << "got=" << estimator.rot() << " expected=" << r;

      ASSERT_TRUE(comparePoints(estimator.t(), t, EPS)) 
        << "got=" << estimator.t() << " expected=" << t;
    }

    void testRegularPointsMono(const cv::Mat& r) {
      MonoReprojectionEstimator estimator(&intrinsics);

      auto pts = regularPoints(r, cv::Point3d());
      auto features = monoFeatures(pts);

      estimator.estimate(features);

      ASSERT_TRUE(compareMats(estimator.rot(), r, EPS))
        << "got=" << estimator.rot() << " expected=" << r;
    }


    StereoIntrinsics intrinsics;
};

TEST_F(ReprojectionEstimatorTest, rotXStereo) {
  testRegularPointsStereo(rotX(10*M_PI/180), cv::Point3d());
}

TEST_F(ReprojectionEstimatorTest, rotYStereo) {
  testRegularPointsStereo(rotY(10*M_PI/180), cv::Point3d());
}

TEST_F(ReprojectionEstimatorTest, rotZStereo) {
  testRegularPointsStereo(rotZ(10*M_PI/180), cv::Point3d());
}

TEST_F(ReprojectionEstimatorTest, rotXMono) {
  testRegularPointsMono(rotX(10*M_PI/180));
}

TEST_F(ReprojectionEstimatorTest, rotYMono) {
  testRegularPointsMono(rotY(10*M_PI/180));
}

TEST_F(ReprojectionEstimatorTest, rotZMono) {
  testRegularPointsMono(rotZ(10*M_PI/180));
}

TEST_F(ReprojectionEstimatorTest, tX) {
  testRegularPointsStereo(cv::Mat::eye(3, 3, CV_64FC1), cv::Point3d(10, 0, 0));
}

TEST_F(ReprojectionEstimatorTest, tY) {
  testRegularPointsStereo(cv::Mat::eye(3, 3, CV_64FC1), cv::Point3d(0, 10, 0));
}

TEST_F(ReprojectionEstimatorTest, tZ) {
  testRegularPointsStereo(cv::Mat::eye(3, 3, CV_64FC1), cv::Point3d(0, 0, 10));
}

TEST_F(ReprojectionEstimatorTest, random) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> angle_d(-M_PI*30/180, M_PI*30/180);
  std::uniform_real_distribution<double> translation_d(0.0, 10.0);

  for (int i = 0; i < 100; ++i) {
    auto angle = std::bind(angle_d, generator);
    auto t = std::bind(translation_d, generator);

    testRegularPointsStereo(
        rotX(angle())*rotY(angle())*rotZ(angle()),
        cv::Point3d(t(), t(), t()));
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
