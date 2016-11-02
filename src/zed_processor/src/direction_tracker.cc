#include <opencv2/core.hpp>

#include "direction_tracker.hpp"
#include "math3d.hpp"
#include "timer.hpp"

DirectionTracker::DirectionTracker(
    const DirectionTrackerSettings& settings,
    const MonoIntrinsics* camera)
    : camera_(camera), estimator_(camera)  {
}

void DirectionTracker::setTarget(const DirectionTarget* target) {
  target_ = target;
}

bool DirectionTracker::process(
    const std::vector<cv::Point2d>& keypoints,
    const cv::Mat& descriptors) {
  Timer timer;
  computeMatches_(keypoints, descriptors);
  timer.mark("match");
  bool res = reconstructOrientation_();
  timer.mark("reconstruct");

  std::cout << "dir_t: " << timer.str() << std::endl;
  return res;
}

void DirectionTracker::computeMatches_(
    const std::vector<cv::Point2d>& keypoints,
    const cv::Mat& descriptors) {


  cv::BFMatcher matcher(cv::NORM_HAMMING, false);
  matches_.resize(0);
  matcher.knnMatch(target_->descriptors, descriptors, matches_, 1);

  
  reprojection_features_.resize(0);
  for (const auto& m : matches_) {
    const auto& cur_kp = keypoints[m[0].trainIdx];
    const auto& target_kp = target_->keypoints[m[0].queryIdx];
    reprojection_features_.push_back(
        MonoReprojectionFeature {
          unproject_(cur_kp),
          unproject_(target_kp),
          cur_kp,
          target_kp
        });
  }


  /* for (int i = 0; i < target_->keypoints.size(); ++i) { */
  /*   const auto& target_kp = target_->keypoints[i]; */
  /*   cv::Mat target_desc = target_->descriptors.row(i); */

  /*   double d1 = 100000, d2 = 100000; */
  /*   int j1 = -1; */

  /*   for (int j = 0; j < keypoints.size(); ++j) { */
  /*     double d = descriptorDist(target_desc, descriptors.row(j)); */
  /*     if (d < d1) { */
  /*       d2 = d1; */
  /*       d1 = d; */
  /*       j1 = j; */
  /*     } else if (d < d2) { */
  /*       d2 = d; */
  /*     } */
  /*   } */

  /*   if (d1 < d2 - 10) { */
  /*     reprojection_features_.push_back( */
  /*       MonoReprojectionFeature { */
  /*         unproject_(keypoints[j1]), */
  /*         unproject_(target_kp), */
  /*         keypoints[j1], */
  /*         target_kp, */ 
  /*       } */
  /*     ); */
  /*   } */
  /* } */
}

cv::Point3d DirectionTracker::unproject_(const cv::Point2d& p) {
  cv::Point3d r;

  r.x = (p.x - camera_->cx)/camera_->f;
  r.y = (p.y - camera_->cy)/camera_->f;
  r.z = 1.0;

  return r;
}

bool DirectionTracker::reconstructOrientation_() {
  rand_.seed(0);

  MonoReprojectionEstimator estimator(camera_);


  best_orientation_ = cv::Mat::eye(3, 3, CV_64FC1);
  int best_inlier_count = 0;

  for (int t = 0; t < settings_.ransac_iterations; ++t) {
    reprojection_features_sample_.resize(settings_.ransac_points);
    // Build ransac points
    for (int i = 0; i < settings_.ransac_points; ++i) {
      int j = std::uniform_int_distribution<int>(
          i, reprojection_features_.size() - 1)(rand_);

      std::swap(reprojection_features_[j], reprojection_features_[i]);
      reprojection_features_sample_[i] = reprojection_features_[i];
    }

    // estimate
    estimator.estimate(reprojection_features_sample_);
    auto r = estimator.rot();
    auto inv = estimator.rot().t();

    // Compute inliers
    reprojection_features_sample_.resize(0);
    for (const auto& f : reprojection_features_) {
      auto s2 = projectPoint(*camera_, r * f.r1);
      auto s1 = projectPoint(*camera_, inv * f.r2);
      
      double err = norm2(s2 - f.s2) + norm2(s1 - f.s1);
      if (err < settings_.ransac_inlier_threshold) {
        reprojection_features_sample_.push_back(f);
      }
    }

    // reestimate
    int n = reprojection_features_sample_.size();
    if (n >= settings_.ransac_min_inliers && n >= best_inlier_count) {
      estimator.estimate(reprojection_features_sample_);
      best_inlier_count = n;
      best_sample_ = reprojection_features_sample_;
      estimator.rot().copyTo(best_orientation_);
    }
  }

  std::cout 
    << "Direction: "
    << "total_matches = " << reprojection_features_.size()
    << "; best_inlier_count = " << best_inlier_count
    << std::endl;

  return best_inlier_count > 0;
}
