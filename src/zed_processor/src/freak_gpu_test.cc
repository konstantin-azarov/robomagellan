#include <chrono>
#include <iostream>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "fast_gpu.hpp"
#include "freak_gpu.hpp"

backward::SignalHandling sh;

const std::string kTestImg = 
    "../../src/zed_processor/test_data/fast_test_image.png";
const float kFeatureSize = 64.980350;

cv::Mat_<uint8_t> linearToOpenCv(cv::Mat_<uint8_t> m) {
  if (m.cols != 512/8) {
    abort();
  }

  cv::Mat_<uint8_t> res = cv::Mat_<uint8_t>::zeros(1, 512/8);

  for (int j = 0; j < 512; ++j) {
    int v = (m[0][j/8] & (1 << (j%8))) ? 1 : 0;
    int j1 = j % 128;
    int b = ( 15 - (j1 % 16)) * 8 + 7 - j1/16 + (j/128)*128;
    res[0][b/8] |= v << (7 - (b%8));
  }

  return res;
}

cv::Mat_<uint8_t> gpuToLinear(cv::Mat_<uint8_t> m) {
  if (m.cols != 512/8) {
    abort();
  }

  cv::Mat_<uint8_t> res = cv::Mat_<uint8_t>::zeros(1, 512/8);
 
  for (int t = 0; t < 64; ++t) {
    for (int j = 0; j < 8; ++j) {
      int v = (m[0][t] & (1 << j)) ? 1 : 0;
      int b = t + j*64;
      res[0][b/8] |= v << (b%8);
      b++;
    }
  }
  return res;
}

TEST(FreakGpu, describe) {
  auto img = cv::imread(kTestImg, cv::IMREAD_GRAYSCALE);
  if (img.data == nullptr) {
    abort();
  }
  cv::cudev::GpuMat_<uint8_t> img_gpu(img);
  cv::cudev::GpuMat_<uint> integral_img;

  cv::cuda::integral(img_gpu, integral_img);

  FreakGpu freak(kFeatureSize);
  FastGpu fast(100000, freak.borderWidth()+5);

  CudaDeviceVector<short3> keypoints_gpu(100000);
  std::vector<short3> keypoints_cpu; 
  fast.detect(img_gpu, 50, keypoints_gpu);
  keypoints_gpu.download(keypoints_cpu);
  std::sort(std::begin(keypoints_cpu), std::end(keypoints_cpu),
      [](const short3& a, const short3& b) {
        return a.y != b.y ? a.y < b.y : a.x < b.x;
      });

  keypoints_gpu.upload(keypoints_cpu);

  cv::cudev::GpuMat_<uint8_t> descriptors_gpu(
      keypoints_cpu.size(), FreakGpu::kDescriptorWidth);
  freak.describe(
      integral_img, keypoints_gpu, keypoints_cpu.size(), descriptors_gpu,
      cv::cuda::Stream::Null());
 
  cv::Mat_<uint8_t> descriptors_cpu;
  descriptors_gpu.download(descriptors_cpu);

  std::vector<cv::KeyPoint> keypoints_opencv(keypoints_cpu.size());
  std::transform(
      std::begin(keypoints_cpu), std::end(keypoints_cpu),
      std::begin(keypoints_opencv),
      [](short3 kp) {
        cv::KeyPoint res;
        res.pt.x = kp.x;
        res.pt.y = kp.y;
        res.response = kp.z;
        return res;
      });

  auto opencv_freak = cv::xfeatures2d::FREAK::create(true, false);
  cv::Mat_<uint8_t> descriptors_opencv;
  opencv_freak->compute(img, keypoints_opencv, descriptors_opencv);

  ASSERT_EQ(descriptors_cpu.rows, descriptors_opencv.rows);
  for (int i=0; i < keypoints_cpu.size(); ++i) {
    bool good = !cv::countNonZero(
        descriptors_opencv.row(i) != 
            linearToOpenCv(gpuToLinear(descriptors_cpu.row(i))));
    
    ASSERT_TRUE(good)
      << "i = " << i << ", pt = [" 
      << keypoints_cpu[i].x << ", " << keypoints_cpu[i].y << "]" << std::endl
      << "opencv = " << descriptors_opencv.row(i) << std::endl
      << "gpu =    " << linearToOpenCv(gpuToLinear(descriptors_cpu.row(i)));
  }
};

class FreakGpuBenchmark : public benchmark::Fixture {
  public:
    FreakGpuBenchmark() : 
        freak_(kFeatureSize),
        keypoints_gpu_(100000) {
    }

    void SetUp(const benchmark::State& s) {
      auto base_img = cv::imread(kTestImg, cv::IMREAD_GRAYSCALE);
      cv::Mat img;
      cv::resize(base_img, img, {s.range(0), s.range(1)});   
      img_gpu_.upload(img);

      cv::cuda::integral(img_gpu_, integral_gpu_);

      FastGpu fast(100000, freak_.borderWidth()+5);

      std::vector<short3> keypoints_cpu; 
      fast.detect(img_gpu_, s.range(2), keypoints_gpu_);
      keypoints_gpu_.download(keypoints_cpu);
      std::sort(std::begin(keypoints_cpu), std::end(keypoints_cpu),
          [](const short3& a, const short3& b) {
            return a.y != b.y ? a.y < b.y : a.x < b.x;
          });

      keypoints_gpu_.upload(keypoints_cpu);
      keypoints_count_ = keypoints_gpu_.size();
      descriptors_gpu_.create(
          keypoints_cpu.size(), FreakGpu::kDescriptorWidth);
    }

  protected:
    FreakGpu freak_;
    cv::cudev::GpuMat_<uint8_t> img_gpu_;
    cv::cudev::GpuMat_<uint> integral_gpu_;
    CudaDeviceVector<short3> keypoints_gpu_;
    int keypoints_count_;
    cv::cudev::GpuMat_<uint8_t> descriptors_gpu_;
};

BENCHMARK_DEFINE_F(FreakGpuBenchmark, describe)(benchmark::State& state) {
  auto& stream = cv::cuda::Stream::Null();
  while (state.KeepRunning()) {
    freak_.describe(
        integral_gpu_, 
        keypoints_gpu_, keypoints_count_, 
        descriptors_gpu_, 
        stream);
    stream.waitForCompletion();
  }

  state.SetItemsProcessed(keypoints_count_);
}

BENCHMARK_REGISTER_F(FreakGpuBenchmark, describe)
  ->Unit(benchmark::kMicrosecond)
  ->Args({4160, 2340, 50})
  ->Args({1280, 720, 25})
  ->Args({1280, 720, 50});

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
