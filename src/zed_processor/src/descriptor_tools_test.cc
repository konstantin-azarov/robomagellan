#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "descriptor_tools.hpp"

backward::SignalHandling sh;

TEST(DescriptorTools, compact) {
  cv::Mat_<uint8_t> descs(5, 64);

  for (int i = 0; i < 5; ++i) {
    for (int j=0; j < 5; ++j) {
      descs(i, j) = i*64 + j;
    }
  }

  std::vector<ushort2> idx = { { 0, 1}, {2, 3} };


  cv::cudev::GpuMat_<uint8_t> descs_gpu(descs);
  cv::cudev::GpuMat_<uint8_t> c1_gpu(2, 64), c2_gpu(2, 64);
  CudaDeviceVector<ushort2> idx_gpu(2);
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

