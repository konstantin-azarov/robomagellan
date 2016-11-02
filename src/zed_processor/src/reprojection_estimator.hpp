#ifndef __REPROJECTION_ESTIMATOR__HPP__
#define __REPROJECTION_ESTIMATOR__HPP__

#include "opencv2/opencv.hpp"

struct StereoIntrinsics;
struct MonoIntrinsics;

struct MonoReprojectionFeature {
  cv::Point3d r1, r2;
  cv::Point2d s1, s2; 
};

struct StereoReprojectionFeature {
  cv::Point3d r1, r2;
  cv::Point2d s1l, s1r, s2l, s2r;
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

    bool estimate(const std::vector<StereoReprojectionFeature>& features);

    const cv::Mat& rot() const { return rot_; }

    const cv::Point3d& t() const { return t_; }
    const cv::Mat& t_cov() const { return t_cov_; }

  private:
    const StereoIntrinsics* intrinsics_;
    cv::Mat rot_;
    cv::Point3d t_;
    cv::Mat t_cov_;
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

    bool estimate(const std::vector<MonoReprojectionFeature>& features);

    const cv::Mat& rot() const { return rot_; }

  private:
    const MonoIntrinsics* intrinsics_;

    cv::Mat rot_;
};

#endif
