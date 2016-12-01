#ifndef __STEREO_MATCHER__HPP__
#define __STEREO_MATCHER__HPP__

#include <opencv2/cudev/ptr2d/gpumat.hpp>

#include "cuda_pinned_allocator.hpp"
#include "cuda_device_vector.hpp"

class Matcher {
  public:
    Matcher(int max_descriptors, int max_pairs);

    // Computes scores and downloads them to the CPU memory
    void computeScores(
        const cv::cudev::GpuMat_<uint8_t>& d1,
        const cv::cudev::GpuMat_<uint8_t>& d2,
        const CudaDeviceVector<ushort2>& pairs_gpu,
        int n_pairs,
        cv::cuda::Stream& stream);

    // Should be called after operations in computeScores complete
     void gatherMatches(
        int n1, int n2,
        PinnedVector<ushort2>& pairs_cpu,
        float threshold_ratio,
        std::vector<cv::Vec2s>& matches);

  private:
    struct Match {
      ushort best, second;
      ushort match;
    };

    // [ 1 x max_pairs ]
    cv::cudev::GpuMat_<uint16_t> scores_gpu_;
    // [ 1 x max_pairs ] 
    cv::Mat_<uint16_t> scores_cpu_; 
    // [ 1 x max_descriptors ]
    std::vector<Match> m1_, m2_;
};

#endif

