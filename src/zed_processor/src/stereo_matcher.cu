#include <cuda_runtime.h>

#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/glob.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#include "stereo_matcher.hpp"

namespace {

  const int kPairsPerBlock = 64;
  const int kThreadsPerPair = 4;

  __global__ void compute_scores(
      const cv::cudev::GlobPtr<uchar> d1,
      const cv::cudev::GlobPtr<uchar> d2,
      const ushort2* pairs,
      int n_pairs,
      ushort* global_scores) {
   
    __shared__ ushort all_scores[kPairsPerBlock+1][kThreadsPerPair];
    volatile ushort* scores = all_scores[threadIdx.y];

    uint pair_id = kPairsPerBlock * blockIdx.x + threadIdx.y;

    if (pair_id < n_pairs) {
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
    scores_cpu_(1, max_pairs),
    m1_(max_descriptors),
    m2_(max_descriptors),
    matches_(max_descriptors) {
}

__host__ const std::vector<cv::Vec2s>& Matcher::match(
    const cv::cudev::GpuMat_<uint8_t>& d1,
    const cv::cudev::GpuMat_<uint8_t>& d2,
    const cv::cudev::GpuMat_<cv::Vec2s>& pairs_gpu,
    const std::vector<cv::Vec2s>& pairs_cpu,
    float threshold_ratio) {
  computeScores(d1, d2, pairs_gpu);
  int n = pairs_cpu.size();
  scores_gpu_.colRange(0, n).download(scores_cpu_.colRange(0, n));
  return gatherMatches(d1.rows, d2.rows, pairs_cpu, threshold_ratio);
}

__host__ void Matcher::computeScores(
    const cv::cudev::GpuMat_<uint8_t>& d1,
    const cv::cudev::GpuMat_<uint8_t>& d2,
    const cv::cudev::GpuMat_<cv::Vec2s>& pairs_gpu) {

  int n_blocks = (pairs_gpu.cols + kPairsPerBlock - 1) / kPairsPerBlock;
  dim3 block_dim(kThreadsPerPair, kPairsPerBlock);

  compute_scores<<<n_blocks, block_dim>>>(
      d1, d2, 
      pairs_gpu.ptr<ushort2>(), pairs_gpu.cols, 
      scores_gpu_.ptr<ushort>());

  cudaSafeCall(cudaGetLastError());
}

#include <iostream>

__host__ const std::vector<cv::Vec2s>& Matcher::gatherMatches(
    int n1, int n2,
    const std::vector<cv::Vec2s>& pairs_cpu,
    float threshold_ratio) {
 
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

    updateMatch(m1_[p[0]], s, p[1]);
    updateMatch(m2_[p[1]], s, p[0]);
  }

  /* std::cout << "m1 = "; */
  /* for (int i=0; i < n1; ++i) { */
  /*   std::cout << "[" << m1_[i].best << ", " */ 
  /*     << m1_[i].second << ", " << m1_[i].match << "] "; */
  /* } */
  /* std::cout << std::endl; */

  matches_.resize(0);
  for (int i = 0; i < n1; ++i) {
    const auto& m1 = m1_[i];
    if (m1.best != 0xFFFF && m1.second * 0.8 > m1.best) {
      const auto& m2 = m2_[m1.match];
      if (m2.second * 0.8 > m2.best && m2.match == i) {
        matches_.push_back(cv::Vec2s(i, m1.match));
      }
    }
  }

  return matches_;
}
