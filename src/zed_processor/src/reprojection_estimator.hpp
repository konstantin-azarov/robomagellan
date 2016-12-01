#ifndef __REPROJECTION_ESTIMATOR__HPP__
#define __REPROJECTION_ESTIMATOR__HPP__

#include <Eigen/Geometry>

#include "opencv2/opencv.hpp"

struct StereoIntrinsics;
struct MonoIntrinsics;

struct MonoReprojectionFeature {
  Eigen::Vector3d r1, r2;
  Eigen::Vector2d s1, s2; 
};

struct StereoReprojectionFeature {
  Eigen::Vector3d r1, r2;
  Eigen::Vector2d s1l, s1r, s2l, s2r;
};

// Find rotation and translation that transforms the first set of points into 
// the second set of features and vise versa. I.e: 
//
//    (s2l, s2r) = project(rot()*p1 + t())
//    (s1l, s1r) = project(inv(rot())*(p2 - t())
class StereoReprojectionEstimator {
  public:
    StereoReprojectionEstimator(
        const StereoIntrinsics* intrinsics);

    bool estimate(
        const std::vector<StereoReprojectionFeature>& features,
        Eigen::Quaterniond& r, 
        Eigen::Vector3d& t,
        Eigen::Matrix3d* t_cov);

  private:
    const StereoIntrinsics* intrinsics_;
};


// Find rotation that transforms the first set of points into the second set
// of points. I.e:
//     
//    s2 = project(rot()*p1)
//    s1 = project(inv(rot())*p2)
class MonoReprojectionEstimator {
  public:
    MonoReprojectionEstimator(
        const MonoIntrinsics* intrinsics);

    bool estimate(
        const std::vector<MonoReprojectionFeature>& features,
        Eigen::Quaterniond& r);

  private:
    const MonoIntrinsics* intrinsics_;
};

#endif
