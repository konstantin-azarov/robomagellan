#include <cuda_runtime.h>

#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/glob.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#include "stereo_matcher.hpp"

namespace {

  const int kPairsPerBlock = 64;
  const int kThreadsPerPair = 4;

  __global__ void computeScoresGpu(
      const cv::cudev::GlobPtr<uchar> d1,
      const cv::cudev::GlobPtr<uchar> d2,
      CudaDeviceVector<ushort2>::Dev pairs,
      ushort* global_scores) {
   
    __shared__ ushort all_scores[kPairsPerBlock+1][kThreadsPerPair];
    volatile ushort* scores = all_scores[threadIdx.y];

    uint pair_id = kPairsPerBlock * blockIdx.x + threadIdx.y;

    if (pair_id < pairs.size()) {
      uint tid = threadIdx.x;

      ushort2 p = pairs[pair_id];
      const uint4* r1 = reinterpret_cast<const uint4*>(d1.row(p.x));
      const uint4* r2 = reinterpret_cast<const uint4*>(d2.row(p.y));

      uint4 b1 = *(r1 + tid);
      uint4 b2 = *(r2 + tid);

      scores[tid] = 
        __popc(b1.x ^ b2.x) + 
        __popc(b1.y ^ b2.y) + 
        __popc(b1.z ^ b2.z) + 
        __popc(b1.w ^ b2.w);

      scores[tid] += scores[tid + 2];
      scores[tid] += scores[tid + 1];

      if (tid == 0) global_scores[pair_id] = scores[0];
    }
  }
}

__host__ Matcher::Matcher(int max_descriptors, int max_pairs) :
    scores_gpu_(1, max_pairs),
    m1_(max_descriptors),
    m2_(max_descriptors) {
  /* scores_cpu_.allocator = cv::cuda::HostMem::getAllocator( */
  /*     cv::cuda::HostMem::PAGE_LOCKED); */ 
  scores_cpu_.create(1, max_pairs);
}

__host__ void Matcher::computeScores(
    const cv::cudev::GpuMat_<uint8_t>& d1,
    const cv::cudev::GpuMat_<uint8_t>& d2,
    const CudaDeviceVector<ushort2>& pairs_gpu,
    int n_pairs,
    cv::cuda::Stream& stream) {

  int n_blocks = (n_pairs + kPairsPerBlock - 1) / kPairsPerBlock;
  dim3 block_dim(kThreadsPerPair, kPairsPerBlock);

  auto cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
  computeScoresGpu<<<n_blocks, block_dim, 0, cuda_stream>>>(
      d1, d2, 
      pairs_gpu,
      scores_gpu_.ptr<ushort>());

  scores_gpu_.colRange(0, n_pairs).download(
      scores_cpu_.colRange(0, n_pairs), stream);

  cudaSafeCall(cudaGetLastError());
}

__host__ void Matcher::gatherMatches(
    int n1, int n2,
    PinnedVector<ushort2>& pairs_cpu,
    float threshold_ratio,
    std::vector<ushort2>& matches) {
 
  for (auto& m : m1_) {
    m.best = m.second = m.match = 0xFFFF;
  }

  for (auto& m : m2_) {
    m.best = m.second = m.match = 0xFFFF;
  }

  /* std::cout << "Scores CPU: " <<  scores_cpu_.colRange(0, pairs_cpu.size()) << std::endl; */

  auto updateMatch = [](Match& m, uint16_t s, uint16_t j) {
    if (s < m.best) {
      m = { s, m.best, j };
    } else if (s < m.second) {
      m.second = s;
    }
  };

  for (int i = 0; i < pairs_cpu.size(); ++i) {
    const auto& p = pairs_cpu[i];
    uint16_t s = scores_cpu_(0, i);

    updateMatch(m1_[p.x], s, p.y);
    updateMatch(m2_[p.y], s, p.x);
  }

  /* std::cout << "m1 = "; */
  /* for (int i=0; i < n1; ++i) { */
  /*   std::cout << "[" << m1_[i].best << ", " */ 
  /*     << m1_[i].second << ", " << m1_[i].match << "] "; */
  /* } */
  /* std::cout << std::endl; */

  matches.resize(0);
  for (int i = 0; i < n1; ++i) {
    const auto& m1 = m1_[i];
    if (m1.best != 0xFFFF && m1.second * 0.8 > m1.best) {
      const auto& m2 = m2_[m1.match];
      if (m2.second * 0.8 > m2.best && m2.match == i) {
        matches.push_back(make_ushort2(i, m1.match));
      }
    }
  }
}
