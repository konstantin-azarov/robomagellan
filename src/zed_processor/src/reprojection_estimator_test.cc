#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "reprojection_estimator.hpp"

#include "math3d.hpp"

const double EPS = 1E-6;

namespace e = Eigen;

class ReprojectionEstimatorTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      intrinsics.f = 100;
      intrinsics.dr = 500;
      intrinsics.cx = 320;
      intrinsics.cy = 200;
    }

    std::vector<std::pair<e::Vector3d, e::Vector3d>> regularPoints(
        e::Affine3d t) {
      std::vector<e::Vector3d> src, transformed;
      src.push_back(e::Vector3d(0, 0, 2));
      src.push_back(e::Vector3d(1, 0, 1));
      src.push_back(e::Vector3d(0, 1, 1));
      src.push_back(e::Vector3d(10, 10, 10));

      std::vector<std::pair<e::Vector3d, e::Vector3d>> res;

      for (auto p : src) {
        res.push_back(std::make_pair(p, t*p));
      }

      return res;
    }

    std::vector<StereoReprojectionFeature> stereoFeatures(
        const std::vector<std::pair<e::Vector3d, e::Vector3d>>& pts) {
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

    e::Vector2d projectMono(const e::Vector3d& pt) {
      return e::Vector2d(
        pt.x()/pt.z() * intrinsics.f + intrinsics.cx,
        pt.y()/pt.z() * intrinsics.f + intrinsics.cy);
    }

    std::vector<MonoReprojectionFeature> monoFeatures(
        const std::vector<std::pair<e::Vector3d, e::Vector3d>>& pts) {
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

    void testRegularPointsStereo(e::Affine3d t0) {
      StereoReprojectionEstimator estimator(&intrinsics);
      
      auto pts = regularPoints(t0);
      auto features = stereoFeatures(pts);
     
      e::Quaterniond q;
      e::Vector3d t;

      estimator.estimate(features, q, t, nullptr);

      auto t1 = e::Translation3d(t)*q;
      ASSERT_TRUE(t0.isApprox(t1, 1E-6))
        << "got=" << t1.linear() << ", " << t1.translation() 
        << " expected=" << t0.linear() << ", " << t0.translation();
    }

    void testRegularPointsMono(const e::Quaterniond& q0) {
      MonoReprojectionEstimator estimator(&intrinsics);

      auto pts = regularPoints(e::Affine3d(q0));
      auto features = monoFeatures(pts);

      e::Quaterniond q;
      estimator.estimate(features, q);

      ASSERT_TRUE(q.isApprox(q0, 1E-6))
        << "got=" << e::Matrix3d(q) << " expected=" << e::Matrix3d(q0);
    }


    StereoIntrinsics intrinsics;
};

TEST_F(ReprojectionEstimatorTest, rotXStereo) {
  testRegularPointsStereo(
      e::Affine3d(e::AngleAxisd(10*M_PI/180, e::Vector3d::UnitX())));
  testRegularPointsStereo(
      e::Affine3d(e::AngleAxisd(-10*M_PI/180, e::Vector3d::UnitX())));
}

TEST_F(ReprojectionEstimatorTest, rotYStereo) {
  testRegularPointsStereo(
      e::Affine3d(e::AngleAxisd(10*M_PI/180, e::Vector3d::UnitY())));
  testRegularPointsStereo(
      e::Affine3d(e::AngleAxisd(-10*M_PI/180, e::Vector3d::UnitY())));
}

TEST_F(ReprojectionEstimatorTest, rotZStereo) {
  testRegularPointsStereo(
      e::Affine3d(e::AngleAxisd(10*M_PI/180, e::Vector3d::UnitZ())));
  testRegularPointsStereo(
      e::Affine3d(e::AngleAxisd(-10*M_PI/180, e::Vector3d::UnitZ())));
}

TEST_F(ReprojectionEstimatorTest, rotXMono) {
  testRegularPointsMono(
      e::Quaterniond(e::AngleAxisd(10*M_PI/180, e::Vector3d::UnitX())));
  testRegularPointsMono(
      e::Quaterniond(e::AngleAxisd(-10*M_PI/180, e::Vector3d::UnitX())));
}

TEST_F(ReprojectionEstimatorTest, rotYMono) {
  testRegularPointsMono(
      e::Quaterniond(e::AngleAxisd(10*M_PI/180, e::Vector3d::UnitY())));
  testRegularPointsMono(
      e::Quaterniond(e::AngleAxisd(-10*M_PI/180, e::Vector3d::UnitY())));
}

TEST_F(ReprojectionEstimatorTest, rotZMono) {
  testRegularPointsMono(
      e::Quaterniond(e::AngleAxisd(10*M_PI/180, e::Vector3d::UnitZ())));
  testRegularPointsMono(
      e::Quaterniond(e::AngleAxisd(-10*M_PI/180, e::Vector3d::UnitZ())));
}

TEST_F(ReprojectionEstimatorTest, tX) {
  testRegularPointsStereo(e::Affine3d(e::Translation3d(e::Vector3d(10, 0, 0))));
}

TEST_F(ReprojectionEstimatorTest, tY) {
  testRegularPointsStereo(e::Affine3d(e::Translation3d(e::Vector3d(0, 10, 0))));
}

TEST_F(ReprojectionEstimatorTest, tZ) {
  testRegularPointsStereo(e::Affine3d(e::Translation3d(e::Vector3d(0, 0, 10))));
}

TEST_F(ReprojectionEstimatorTest, random) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> angle_d(-M_PI*30/180, M_PI*30/180);
  std::uniform_real_distribution<double> translation_d(0.0, 10.0);

  for (int i = 0; i < 100; ++i) {
    auto angle = std::bind(angle_d, generator);
    auto t = std::bind(translation_d, generator);

    testRegularPointsStereo(
        e::Translation3d(e::Vector3d(t(), t(), t()))*
        e::AngleAxisd(angle(), e::Vector3d::UnitX())*
        e::AngleAxisd(angle(), e::Vector3d::UnitY())*
        e::AngleAxisd(angle(), e::Vector3d::UnitZ()));
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
