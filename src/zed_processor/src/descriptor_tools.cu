#include <cuda_runtime.h>

#include "descriptor_tools.hpp"

namespace descriptor_tools {
  const int kCopiesPerBlock = 32;

  __global__ void gatherGpu(
      const cv::cudev::GlobPtr<uchar> d1,
      const cv::cudev::GlobPtr<uchar> d2,
      CudaDeviceVector<ushort2>::Dev matches,
      cv::cudev::GlobPtr<uchar> d1_out,
      cv::cudev::GlobPtr<uchar> d2_out) {
    int match_id = kCopiesPerBlock * blockIdx.x + threadIdx.y;

    if (match_id < matches.size()) {
      ushort2 idx = matches[match_id];
      int tid = threadIdx.x;

      const ushort* r1 = reinterpret_cast<const ushort*>(d1.row(idx.x));
      const ushort* r2 = reinterpret_cast<const ushort*>(d2.row(idx.y));

      reinterpret_cast<ushort*>(d1_out.row(match_id))[tid] = r1[tid];
      reinterpret_cast<ushort*>(d2_out.row(match_id))[tid] = r2[tid];
    }
  }

  inline int divUp(int x, int y) {
    return (x + y - 1)/y;
  }

  void gatherDescriptors(
    const cv::cudev::GpuMat_<uint8_t>& d1,
    const cv::cudev::GpuMat_<uint8_t>& d2,
    const CudaDeviceVector<ushort2>& matches,
    int n_matches,
    cv::cudev::GpuMat_<uint8_t>& d1_compact,
    cv::cudev::GpuMat_<uint8_t>& d2_compact,
    cv::cuda::Stream& stream) {

   auto s = cv::cuda::StreamAccessor::getStream(stream);

   dim3 blockDim(32, kCopiesPerBlock);

   gatherGpu<<<divUp(n_matches, kCopiesPerBlock), blockDim, 0, s>>>(
       d1, d2, matches, d1_compact, d2_compact);

   cudaSafeCall(cudaGetLastError());
  }
}
