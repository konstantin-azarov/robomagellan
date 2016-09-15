#ifndef __RIGID_ESTIMATOR_H__
#define __RIGID_ESTIMATOR_H__

#include <opencv2/opencv.hpp>
#include <vector>


// Find rotation and translation that translates first set of points into 
// the second. I.e: 
//
//    p2 = rot()*p1 + t()
class RigidEstimator {
  public:
    RigidEstimator();

    void estimate(
        const std::vector<cv::Point3d>& p1,
        const std::vector<cv::Point3d>& p2);

    const cv::Mat& rot() const { return rot_; }
    const cv::Point3d& t() const { return t_; }
  
  private:
    cv::Point3d center(std::vector<cv::Point3d>& pts);

  private:
    cv::Mat rot_;
    cv::Point3d t_;
    std::vector<cv::Point3d> pts1_, pts2_;
};

#endif
