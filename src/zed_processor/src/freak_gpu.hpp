#ifndef __FREAK_GPU__HPP__
#define __FREAK_GPU__HPP__

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#include "freak_base.hpp"

class FreakGpu : public FreakBase {
  public:
    FreakGpu(double feature_size);

    static bool initialize();

    void describe(
        const cv::cuda::GpuMat& img,
        const cv::cuda::GpuMat& keypoints);

    const cv::cudev::GpuMat& descriptors() const { return descriptors_; }

    cv::Mat descriptorsCpu() {
      cv::Mat res;
      descriptors_.download(res);
      return res;
    }

  private:
    static bool initialized_;
    cv::cudev::GpuMat_<uint> integral_;
    cv::cuda::GpuMat descriptors_;
};

#endif
