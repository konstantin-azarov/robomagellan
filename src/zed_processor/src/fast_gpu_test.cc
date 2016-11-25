#include <algorithm>
#include <chrono>
#include <iostream>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "fast_gpu.hpp"

backward::SignalHandling sh;
  
std::string kTestImg = "src/zed_processor/test_data/fast_test_image.png";
const int kMaxKeypoints = 200000;

TEST(FastGpu, features) {
  const int threshold = 50;

  auto img = cv::imread(kTestImg, cv::IMREAD_GRAYSCALE);
  cv::cudev::GpuMat_<uint8_t> img_gpu(img);

  FastGpu my_fast(kMaxKeypoints, 3);

  std::vector<cv::KeyPoint> opencv_keypoints;

  cv::FAST(img, opencv_keypoints, threshold, true);
 
  my_fast.detect(img_gpu, threshold);
  std::vector<cv::Vec3s> my_keypoints;
  my_fast.keypoints().download(my_keypoints);

  std::sort(std::begin(my_keypoints), std::end(my_keypoints),
      [](const cv::Vec3s& a, const cv::Vec3s& b) {
        return a[1] != b[1] ? a[1] < b[1] : a[0] < b[0];
      });

  std::sort(std::begin(opencv_keypoints), std::end(opencv_keypoints),
      [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        return a.pt.y != b.pt.y ? a.pt.y < b.pt.y : a.pt.x < b.pt.x;
      });

  ASSERT_EQ(opencv_keypoints.size(), my_keypoints.size());
  for (int i = 0; i < my_keypoints.size(); ++i) {
    ASSERT_EQ(opencv_keypoints[i].pt.x, my_keypoints[i][0]) << i;
    ASSERT_EQ(opencv_keypoints[i].pt.y, my_keypoints[i][1]) << i;
    ASSERT_EQ(opencv_keypoints[i].response, my_keypoints[i][2]) << i;
  }
}

class FastBenchmark : public benchmark::Fixture {
  public:
    FastBenchmark() : my_fast_(kMaxKeypoints, 3) {
      base_img_ = cv::imread(kTestImg, cv::IMREAD_GRAYSCALE);
    }

    void SetUp(const benchmark::State& s) {
      cv::Mat img;
      cv::resize(base_img_, img, {s.range(0), s.range(1)});   
      img_gpu_.upload(img);
      threshold_ = s.range(2);

      opencv_fast_ = cv::cuda::FastFeatureDetector::create(
          threshold_, true, cv::FastFeatureDetector::TYPE_9_16, kMaxKeypoints);
    }

  protected:
    FastGpu my_fast_;
    cv::Ptr<cv::cuda::FastFeatureDetector> opencv_fast_;
    cv::Mat base_img_;
    cv::cudev::GpuMat_<uint8_t> img_gpu_;
    cv::cudev::GpuMat keypoints_gpu_;
    int threshold_;
};

BENCHMARK_DEFINE_F(FastBenchmark, openCV)(benchmark::State& state) {
  auto stream = cv::cuda::Stream::Null();
  while (state.KeepRunning()) {
    opencv_fast_->detectAsync(img_gpu_, keypoints_gpu_, cv::noArray(), stream);
    stream.waitForCompletion();
  }
}

BENCHMARK_DEFINE_F(FastBenchmark, mine)(benchmark::State& state) {
  auto stream = cv::cuda::Stream::Null();
  while (state.KeepRunning()) {
    my_fast_.detect(img_gpu_, threshold_);
    cudaDeviceSynchronize();
  }
}

void fastBenchmarkArgs(benchmark::internal::Benchmark* b) {
  b->Unit(benchmark::kMicrosecond);
  b->Args({4160, 2340, 50}); // Original size
  b->Args({4160, 2340, 250}); 
  b->Args({1280, 720, 25});
  b->Args({1280, 720, 50});
  b->Args({1280, 720, 100});
  b->Args({1280, 720, 200});
  b->Args({1280, 720, 250});
  b->Args({640, 480, 25});
  b->Args({320, 200, 25});
}

BENCHMARK_REGISTER_F(FastBenchmark, openCV)->Apply(fastBenchmarkArgs);
BENCHMARK_REGISTER_F(FastBenchmark, mine)->Apply(fastBenchmarkArgs);

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
