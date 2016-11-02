#ifndef __FREAK__HPP__
#define __FREAK__HPP__

#include <opencv2/core.hpp>

#include "freak_base.hpp"

class Freak : public FreakBase {
  public:
    Freak(double feature_size);

    const cv::Mat& describe(
        const cv::Mat& img, 
        std::vector<cv::KeyPoint>& keypoints);

  private:
    void computePoints_(
        const cv::Point2f& center, PatternPoint* points, int* res);

  private:
    cv::Mat_<uint8_t> descriptors_;
    cv::Mat integral_;
};

#endif
