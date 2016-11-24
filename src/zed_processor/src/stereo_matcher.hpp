#ifndef __STEREO_MATCHER__HPP__
#define __STEREO_MATCHER__HPP__

#include <opencv2/cudev/ptr2d/gpumat.hpp>

class Matcher {
  public:
    Matcher(int max_descriptors, int max_pairs);

    const std::vector<cv::Vec2s>& match(
        const cv::cudev::GpuMat_<uint8_t>& d1,
        const cv::cudev::GpuMat_<uint8_t>& d2,
        const cv::cudev::GpuMat_<cv::Vec2s>& pairs_gpu,
        const std::vector<cv::Vec2s>& pairs_cpu,
        float threshold_ratio);

    void computeScores(
        const cv::cudev::GpuMat_<uint8_t>& d1,
        const cv::cudev::GpuMat_<uint8_t>& d2,
        const cv::cudev::GpuMat_<cv::Vec2s>& pairs_gpu);

    const std::vector<cv::Vec2s>& gatherMatches(
        int n1, int n2,
        const std::vector<cv::Vec2s>& pairs_cpu,
        float threshold_ratio);

  private:
    struct Match {
      ushort best, second;
      ushort match;
    };

    cv::cudev::GpuMat_<uint16_t> scores_gpu_;
    cv::Mat_<uint16_t> scores_cpu_; 
    std::vector<Match> m1_, m2_;
    std::vector<cv::Vec2s> matches_;
};

#endif

