#ifndef __DESCRIPTOR_TOOLS__HPP__
#define __DESCRIPTOR_TOOLS__HPP__

#include <opencv2/core.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>
#include "cuda_device_vector.hpp"

namespace descriptor_tools {
  void gatherDescriptors(
    const cv::cudev::GpuMat_<uint8_t>& d1,
    const cv::cudev::GpuMat_<uint8_t>& d2,
    const CudaDeviceVector<ushort2>& matches,
    int n_matches,
    cv::cudev::GpuMat_<uint8_t>& d1_compact,
    cv::cudev::GpuMat_<uint8_t>& d2_compact,
    cv::cuda::Stream& stream);

  void gatherDescriptors(
    const CudaDeviceVector<const uint8_t*>& descs,
    int n,
    cv::cudev::GpuMat_<uint8_t>& d_compact,
    cv::cuda::Stream& stream);

  void scores(
     const cv::cudev::GpuMat_<uint8_t>& d1,
     const cv::cudev::GpuMat_<uint8_t>& d2,
     cv::cudev::GpuMat_<uint16_t>& scores,
     cv::cuda::Stream& stream);
};

#endif
