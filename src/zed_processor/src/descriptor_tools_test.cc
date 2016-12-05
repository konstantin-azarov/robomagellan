#include <iostream>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "descriptor_tools.hpp"

backward::SignalHandling sh;

TEST(DescriptorTools, compact) {
  const int n = 10;
  cv::Mat_<uint8_t> descs(10, 64);

  for (int i = 0; i < n; ++i) {
    for (int j=0; j < 64; ++j) {
      descs(i, j) = i*64 + j;
    }
  }

  std::vector<ushort2> idx = { { 0, 1}, {4, 3}, {2, 1}, {4, 2}, {5, 1}, {1, 5} };


  cv::cudev::GpuMat_<uint8_t> descs_gpu(descs);
  cv::cudev::GpuMat_<uint8_t> c1_gpu(idx.size(), 64), c2_gpu(idx.size(), 64);
  CudaDeviceVector<ushort2> idx_gpu(idx.size());
  idx_gpu.upload(idx);
  descriptor_tools::gatherDescriptors(
      cv::cudev::GpuMat_<uint8_t>(descs), 
      cv::cudev::GpuMat_<uint8_t>(descs), 
      idx_gpu, idx.size(), 
      c1_gpu, c2_gpu,
      cv::cuda::Stream::Null());

  cv::Mat_<uint8_t> c1, c2;
  c1_gpu.download(c1);
  c2_gpu.download(c2);

  for (int i=0; i < idx.size(); ++i) {
    for (int j=0; j < 64; ++j) {
      ASSERT_EQ(descs(idx[i].x, j), c1(i, j)) << i << ", " << j;
      ASSERT_EQ(descs(idx[i].y, j), c2(i, j)) << i << ", " << j;
    }
  }
}

cv::Mat_<uchar> randomDescriptors(int n) {
  cv::Mat_<uchar> res(n, 64);

  std::default_random_engine rand(0);
  std::uniform_int_distribution<> dist(0, 0xFFFF);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 64; ++j) {
      res[i][j] = dist(rand);
    }
  }

  return res;
}

TEST(DescriptorTools, pairwiseScores) {
  auto d1 = randomDescriptors(20);
  auto d2 = randomDescriptors(20);

  cv::Mat_<ushort> golden_scores(d1.rows, d2.rows);

  for (int i = 0; i < d1.rows; ++i) {
    for (int j = 0; j < d2.rows; ++j) {
      golden_scores(i, j) = cv::norm(d1.row(i), d2.row(j), cv::NORM_HAMMING); 
    }
  }

  cv::cudev::GpuMat_<uint8_t> d1_gpu(d1), d2_gpu(d2);
  cv::cudev::GpuMat_<ushort> scores_gpu(d1.rows, d2.rows);

  descriptor_tools::scores(
      d1_gpu, d2_gpu, scores_gpu, cv::cuda::Stream::Null());

  cv::Mat_<ushort> scores;
  scores_gpu.download(scores);

  ASSERT_EQ(0, cv::countNonZero(scores != golden_scores))
    << golden_scores << std::endl << scores;
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "benchmark") {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
  } else {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }
}

