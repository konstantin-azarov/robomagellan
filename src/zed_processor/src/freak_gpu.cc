#include <iostream>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>
#include <opencv2/cudaarithm.hpp>

#include "freak_gpu.hpp"

namespace freak_gpu {
  extern void describeKeypointsGpu(
      const cv::cudev::GlobPtr<uint> img,
      const short3* keypoints,
      int num_keypoints,
      cv::cuda::PtrStepSzb descriptors);

  extern bool initialize(
      float3* points, int num_points,
      short4* orientation_pairs, int num_orientation_pairs,
      short2* descriptor_pairs, int num_descriptor_pairs);
};

bool FreakGpu::initialized_ = false;

FreakGpu::FreakGpu(double feature_size) : FreakBase(feature_size) {
  if (!initialized_) {
    initialized_ = true;

    std::vector<float3> points;
    std::vector<short4> orientation_pairs;
    std::vector<short2> descriptor_pairs;

    for (const auto& p : patterns_) {
      points.push_back(make_float3(p.x, p.y, p.sigma));
    }

    for (const auto& p : orientation_pairs_) {
      if (p.dx > (1 << 15) || p.dx < -(1 << 15) ||
          p.dy > (1 << 15) || p.dy < -(1 << 15)) {
        abort();
      }
      orientation_pairs.push_back(make_short4(p.i, p.j, p.dx, p.dy));
    }

    for (const auto& p : descriptor_pairs_) {
      descriptor_pairs.push_back(make_short2(p.i, p.j));
    }

    if (!freak_gpu::initialize(
          points.data(), points.size(),
          orientation_pairs.data(), orientation_pairs.size(),
          descriptor_pairs.data(), descriptor_pairs.size())) {
      abort();
    }
  }
}

void FreakGpu::describe(
    const cv::cuda::GpuMat& img,
    const cv::cuda::GpuMat& keypoints) {

  cv::cuda::integral(img, integral_);

  descriptors_.create(keypoints.cols, 512/8, CV_8UC1);
  descriptors_.setTo(cv::Scalar(0));

  freak_gpu::describeKeypointsGpu(
      integral_, 
      keypoints.ptr<short3>(), 
      keypoints.cols,
      descriptors_);
}
