#ifndef __FREAK_GPU__HPP__
#define __FREAK_GPU__HPP__

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#include "cuda_device_vector.hpp"
#include "freak_base.hpp"

class FreakGpu : public FreakBase {
  public:
    static const int kDescriptorWidth = 64; // bytes

    // It has to be private because CUDA wants to typedef it
    struct FreakConsts {
      float3* points;
      short4* orientation_pairs;
      short2* descriptor_pairs;
    };
  public:
    FreakGpu(double feature_size);
    ~FreakGpu();

    static bool initialize();

    void describe(
        const cv::cudev::GpuMat_<uint>& integral_img,
        const CudaDeviceVector<short3>& keypoints,
        int keypoints_count,
        cv::cudev::GpuMat_<uint8_t>& descriptors,
        cv::cuda::Stream& stream);
 
#ifdef __CUDACC__
    friend __global__ void describeKeypoints(
        FreakConsts consts,
        const cv::cudev::GlobPtr<uint> integral_img,
        const CudaDeviceVector<short3>::Dev keypoints,
        cv::cuda::PtrStepSzb descriptors);
    
    friend __device__ float computePoint(
        const FreakConsts consts,
        const cv::cudev::GlobPtr<uint> integral_img,
        short3 center,
        int pt_index);
#endif

  private:
    FreakConsts consts_;
};

#endif
