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

#include "cuda_device_vector.hpp"
#include "fast_gpu.hpp"

backward::SignalHandling sh;
  
const std::string kTestImg = 
    "../../src/zed_processor/test_data/fast_test_image.png";
const int kMaxKeypoints = 200000;

TEST(FastGpu, features) {
  const int threshold = 50;

  auto img = cv::imread(kTestImg, cv::IMREAD_GRAYSCALE);
  if (img.data == nullptr) {
    abort();
  }
  cv::cudev::GpuMat_<uint8_t> img_gpu(img);

  FastGpu my_fast(kMaxKeypoints, 3);

  std::vector<cv::KeyPoint> opencv_keypoints;

  cv::FAST(img, opencv_keypoints, threshold, true);
 
  auto s = cv::cuda::Stream::Null();

  CudaDeviceVector<short3> my_keypoints_dev(kMaxKeypoints);
 
  my_fast.computeScores(img_gpu, threshold, s);
  my_fast.downloadKpCount(s);
  s.waitForCompletion();
  my_fast.extract(threshold, my_keypoints_dev, s);
  s.waitForCompletion();

  std::vector<short3> my_keypoints;
  my_keypoints_dev.download(my_keypoints);

  std::sort(std::begin(my_keypoints), std::end(my_keypoints),
      [](const short3& a, const short3& b) {
        return a.y != b.y ? a.y < b.y : a.x < b.x;
      });

  std::sort(std::begin(opencv_keypoints), std::end(opencv_keypoints),
      [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        return a.pt.y != b.pt.y ? a.pt.y < b.pt.y : a.pt.x < b.pt.x;
      });

  ASSERT_EQ(opencv_keypoints.size(), my_keypoints.size());
  for (int i = 0; i < my_keypoints.size(); ++i) {
    ASSERT_EQ(opencv_keypoints[i].pt.x, my_keypoints[i].x) << i;
    ASSERT_EQ(opencv_keypoints[i].pt.y, my_keypoints[i].y) << i;
    ASSERT_EQ(opencv_keypoints[i].response, my_keypoints[i].z) << i;
  }
}

class FastBenchmark : public benchmark::Fixture {
  public:
    FastBenchmark() : 
        my_fast_(kMaxKeypoints, 3), 
        keypoints_gpu_(kMaxKeypoints) {
    }

    void SetUp(const benchmark::State& s) {
      auto base_img = cv::imread(kTestImg, cv::IMREAD_GRAYSCALE);
      cv::Mat img;
      cv::resize(base_img, img, {s.range(0), s.range(1)});   
      img_gpu_.upload(img);
      threshold_ = s.range(2);

      opencv_fast_ = cv::cuda::FastFeatureDetector::create(
          threshold_, true, cv::FastFeatureDetector::TYPE_9_16, kMaxKeypoints);
    }

  protected:
    FastGpu my_fast_;
    cv::Ptr<cv::cuda::FastFeatureDetector> opencv_fast_;
    cv::cudev::GpuMat_<uint8_t> img_gpu_;
    cv::cuda::GpuMat keypoints_opencv_;
    CudaDeviceVector<short3> keypoints_gpu_;
    int threshold_;

    cv::cuda::Stream s_;
    cv::cuda::Event start_, end_;
};

BENCHMARK_DEFINE_F(FastBenchmark, openCv)(benchmark::State& state) {
  auto stream = cv::cuda::Stream::Null();
  while (state.KeepRunning()) {
    opencv_fast_->detectAsync(
        img_gpu_, keypoints_opencv_, cv::noArray(), stream);
    stream.waitForCompletion();
  }
}

BENCHMARK_DEFINE_F(FastBenchmark, mine)(benchmark::State& state) {
  while (state.KeepRunning()) {
    keypoints_gpu_.clear();
    my_fast_.computeScores(img_gpu_, threshold_, s_);
    s_.waitForCompletion();
    my_fast_.extract(threshold_, keypoints_gpu_, s_);
    s_.waitForCompletion();
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

BENCHMARK_REGISTER_F(FastBenchmark, openCv)->Apply(fastBenchmarkArgs);
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
