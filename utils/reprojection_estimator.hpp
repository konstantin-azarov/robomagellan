#ifndef __REPROJECTION_ESTIMATOR__HPP__
#define __REPROJECTION_ESTIMATOR__HPP__

#include "opencv2/opencv.hpp"

struct StereoIntrinsics;

struct ReprojectionFeature {
  cv::Point3d r1, r2;
  cv::Point2d s1l, s1r, s2l, s2r;
};

class ReprojectionEstimator {
  public:
    ReprojectionEstimator(const StereoIntrinsics* intrinsics);

    void estimate(const std::vector<ReprojectionFeature>& features);

    const cv::Mat& rot() const { return rot_; }
    const cv::Point3d& t() const { return t_; }

  private:
    const StereoIntrinsics* intrinsics_;
    cv::Mat rot_;
    cv::Point3d t_;
};

#endif
