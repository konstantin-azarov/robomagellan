#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "reprojection_estimator.hpp"

#include "math3d.hpp"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Covariance;
using ceres::Problem;
using ceres::Solver;
using ceres::QuaternionParameterization;
using ceres::UnitQuaternionRotatePoint;
using ceres::QuaternionToRotation;
using ceres::Solve;

template <class T>
struct ForwardTransform {
  static void t(const T* p0, const T* const q, const T* const t, T* p) {
    T p1[3];

    UnitQuaternionRotatePoint(q, p0, p1);

    p[0] = p1[0] + t[0];
    p[1] = p1[1] + t[1];
    p[2] = p1[2] + t[2];
  }
};

template <class T>
struct ReverseTransform {
  static void t(const T* p0, const T* const q, const T* const t, T* p) {
    T q1[4] = { q[0], -q[1], -q[2], -q[3]};
    T p1[3] = { p0[0] - t[0], p0[1] - t[1], p0[2] - t[2] };

    UnitQuaternionRotatePoint(q1, p1, p);
  }
};

template <template <class T> class Transform> 
class ReprojectionError {
  public:
    ReprojectionError(const cv::Point3d& r, 
                      const cv::Point2d& sl, 
                      const cv::Point2d& sr,
                      const StereoIntrinsics* camera) : 
      r_(r), sl_(sl), sr_(sr), c_(camera) {
    }

    template <class T>
    bool operator()(const T* const q, const T* const t, T* e) const {
      T p0[3] = { T(r_.x), T(r_.y), T(r_.z) };
      T p[3];

      Transform<T>::t(p0, q, t, p);

      // Project
      T v = T(c_->f) * p[1] / p[2] + T(c_->cy);

      e[0] = T(c_->f) * p[0] / p[2] + T(c_->cxl - sl_.x);
      e[1] = v - T(sl_.y);
      e[2] = (T(c_->f) * p[0] + T(c_->dr)) / p[2] + T(c_->cxr - sr_.x);
      e[3] = v - T(sr_.y);

      return true;
    }

  private:
    cv::Point3d r_;
    cv::Point2d sl_, sr_;
    const StereoIntrinsics* c_;
};

using ForwardResidual = ReprojectionError<ForwardTransform>;
using ReverseResidual = ReprojectionError<ReverseTransform>;

ReprojectionEstimator::ReprojectionEstimator(const StereoIntrinsics* intrinsics) :
  intrinsics_(intrinsics) {
}

bool ReprojectionEstimator::estimate(
    const std::vector<ReprojectionFeature>& features) {

  double q[4] = { 1, 0, 0, 0 };
  double t[3] = { 0, 0, 0 };
  std::vector<double*> allParameterBlocks = { q, t };

  Problem problem;

  problem.AddParameterBlock(q, 4, new QuaternionParameterization());
  problem.AddParameterBlock(t, 3);

  for (auto f : features) {
    problem.AddResidualBlock(
        new AutoDiffCostFunction<ForwardResidual, 4, 4, 3>(
          new ForwardResidual(f.r1, f.s2l, f.s2r, intrinsics_)),
        new ceres::SoftLOneLoss(1.0),
        allParameterBlocks);

    problem.AddResidualBlock(
        new AutoDiffCostFunction<ReverseResidual, 4, 4, 3>(
          new ReverseResidual(f.r2, f.s1l, f.s1r, intrinsics_)),
        new ceres::SoftLOneLoss(1.0),
        allParameterBlocks);
  }

  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::vector<std::pair<const double*, const double*> > covariance_blocks;
  covariance_blocks.push_back(std::make_pair(t, t));

  t_cov_.create(3, 3, CV_64F);
  
  Covariance::Options cov_options;
  Covariance covariance(cov_options);

  if (covariance.Compute(covariance_blocks, &problem)) {
    covariance.GetCovarianceBlock(t, t, reinterpret_cast<double*>(t_cov_.data));
  } else {
    t_cov_ = cv::Mat::zeros(3, 3, CV_64F);
  }
  

  rot_.create(3, 3, CV_64F);
  QuaternionToRotation(q, reinterpret_cast<double*>(rot_.data));

  t_.x = t[0];
  t_.y = t[1];
  t_.z = t[2];
  
  return true;
}
