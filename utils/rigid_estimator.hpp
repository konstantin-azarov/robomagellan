#ifndef __RIGID_ESTIMATOR_H__
#define __RIGID_ESTIMATOR_H__

#include <opencv2/opencv.hpp>
#include <vector>

class RigidEstimator {
  public:
    RigidEstimator();

    void estimate(const std::vector<std::pair<cv::Point3d, cv::Point3d> >& points);

    const cv::Mat& rot() const { return rot_; }
    const cv::Point3d& t() const { return t_; }
  
  private:
    void unpack(const std::vector<std::pair<cv::Point3d, cv::Point3d> >& points);
    cv::Point3d center(std::vector<cv::Point3d>& pts);

  private:
    cv::Mat rot_;
    cv::Point3d t_;
    std::vector<cv::Point3d> pts1_, pts2_;
};

#endif
