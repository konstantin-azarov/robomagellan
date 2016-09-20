#ifndef __REPROJECTION_ESTIMATOR__HPP__
#define __REPROJECTION_ESTIMATOR__HPP__

#include "opencv2/opencv.hpp"

struct StereoIntrinsics;

struct ReprojectionFeature {
  cv::Point3d r1, r2;
  cv::Point2d s1l, s1r, s2l, s2r;
};

// Find rotation and translation that translates first set of points into 
// the second set of features and vise versa. I.e: 
//
//    (s2l, s2r) = project(rot()*p1 + t())
//    (s1l, s1r) = project(inv(rot())*(p2 - t())
class ReprojectionEstimator {
  public:
    ReprojectionEstimator(const StereoIntrinsics* intrinsics);

    bool estimate(const std::vector<ReprojectionFeature>& features);

    const cv::Mat& rot() const { return rot_; }
    const cv::Point3d& t() const { return t_; }
    const cv::Mat& t_cov() const { return t_cov_; }

  private:
    const StereoIntrinsics* intrinsics_;
    cv::Mat rot_;
    cv::Point3d t_;
    cv::Mat t_cov_;
};

#endif
