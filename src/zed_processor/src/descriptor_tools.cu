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

  const int kDescsPerBlock = 8;
  const int kThreadsPerPair = 4;

  __global__ void matchGpu(
      const cv::cudev::GlobPtr<uchar> d1,
      const cv::cudev::GlobPtr<uchar> d2,
      int n1, int n2,
      cv::cudev::GlobPtr<ushort> global_scores) {

    __shared__ ushort all_scores[kDescsPerBlock+1][kDescsPerBlock][kThreadsPerPair];
    volatile ushort* scores = all_scores[threadIdx.y][threadIdx.z];

    int i1 = blockDim.y * blockIdx.x + threadIdx.y;
    int i2 = blockDim.z * blockIdx.y + threadIdx.z;

    if (i1 < n1 && i2 < n2) {
      uint tid = threadIdx.x;
     

      uint4 b1 = reinterpret_cast<const uint4*>(d1.row(i1))[tid];
      uint4 b2 = reinterpret_cast<const uint4*>(d2.row(i2))[tid];
   
      scores[tid] = 
        __popc(b1.x ^ b2.x) + 
        __popc(b1.y ^ b2.y) + 
        __popc(b1.z ^ b2.z) + 
        __popc(b1.w ^ b2.w);
      
      scores[tid] += scores[tid + 2];
      scores[tid] += scores[tid + 1];

      if (tid == 0) {
        global_scores(i1, i2) = scores[0];
      }
    }
  }

  void scores(
      const cv::cudev::GpuMat_<uint8_t>& d1,
      const cv::cudev::GpuMat_<uint8_t>& d2,
      cv::cudev::GpuMat_<uint16_t>& scores,
      cv::cuda::Stream& stream) {

    int n1 = d1.rows;
    int n2 = d2.rows;
   
    auto s = cv::cuda::StreamAccessor::getStream(stream);

    dim3 block_dim(kThreadsPerPair, kDescsPerBlock, kDescsPerBlock);
    dim3 grid_dim(divUp(n1, block_dim.y), divUp(n2, block_dim.z));

    matchGpu<<<grid_dim, block_dim, 0, s>>>(d1, d2, n1, n2, scores);
   
    cudaSafeCall(cudaGetLastError());
  }
}
