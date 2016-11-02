#ifndef __DIRECTION_TRACKER__HPP__
#define __DIRECTION_TRACKER__HPP__

#include <vector>
#include <opencv2/core.hpp>
#include <random>

#include "direction_target.hpp"
#include "reprojection_estimator.hpp"

struct Match {
  Match() {}

  Match(const cv::Point2f& p1, const cv::Point2f& p2) 
      : target(p1), current(p2) {
  }

  cv::Point2f target, current;
};

struct DirectionTrackerSettings {
  int match_threshold = 10;
  
  int ransac_points = 3;
  int ransac_iterations = 100;
  double ransac_inlier_threshold = 3.0;
  int ransac_min_inliers = 10;
};

class DirectionTracker {
  public:
    DirectionTracker(
        const DirectionTrackerSettings& settings,
        const MonoIntrinsics* camera);

    void setTarget(const DirectionTarget* target);

    bool process(
        const std::vector<cv::Point2d>& keypoints,
        const cv::Mat& descriptors);

    const cv::Mat& rot() const { return best_orientation_; }

    const std::vector<MonoReprojectionFeature>& reprojectionFeatures() const {
      return reprojection_features_;
    }

    const std::vector<MonoReprojectionFeature>& bestSample() const {
      return best_sample_;
    }

    const DirectionTarget* target() const {
      return target_;
    }

  private:
    void computeMatches_(
        const std::vector<cv::Point2d>& keypoints,
        const cv::Mat& descriptors);

    cv::Point3d unproject_(const cv::Point2d& p);

    bool reconstructOrientation_();

  private:
    DirectionTrackerSettings settings_;
    const MonoIntrinsics* camera_;

    MonoReprojectionEstimator estimator_;
    cv::Mat best_orientation_;

    const DirectionTarget* target_;
    std::vector<MonoReprojectionFeature> reprojection_features_;
    std::vector<MonoReprojectionFeature> reprojection_features_sample_;
    std::vector<MonoReprojectionFeature> best_sample_;
    std::vector<std::vector<cv::DMatch>> matches_;

    std::default_random_engine rand_;
};

#endif

