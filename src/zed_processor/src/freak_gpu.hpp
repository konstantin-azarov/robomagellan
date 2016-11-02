#ifndef __FREAK_GPU__HPP__
#define __FREAK_GPU__HPP__

#include <opencv2/core.hpp>

#include "freak_base.hpp"

class FreakGpu : public FreakBase {
  public:
    FreakGpu(double feature_size);

    static bool initialize();

    cv::Mat describe(
        const cv::Mat& img,
        const std::vector<cv::KeyPoint>& keypoints); 

  private:
    static bool initialized_;
};

#endif
