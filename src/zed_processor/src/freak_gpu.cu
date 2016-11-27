#include <stdio.h>

#include <cuda_runtime.h>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/glob.hpp>

#include "freak_gpu.hpp"

const int kNumThreads = 64;

const int kKeyPointsPerBlock = 2;

namespace {
  __device__ float bad_floorf(float f) {
    if (f >= 0) {
      return floorf(f);
    } else {
      return -floorf(-f);
    }
  }

  __device__ float2 reduce(float2* data) {
    uint tid = threadIdx.x;

    for (uint s = blockDim.x/2; s > 0; s /= 2) {
      if (tid < s) {
        data[tid].x += data[tid + s].x;
        data[tid].y += data[tid + s].y;
      }

      __syncthreads();
    }

    return data[0];
  }

}

__device__ float computePoint(
    const FreakGpu::FreakConsts consts,
    const cv::cudev::GlobPtr<uint> integral_img,
    short3 center,
    int pt_index) {
  float cx = center.x + consts.points[pt_index].x;
  float cy = center.y + consts.points[pt_index].y;
  float sigma = consts.points[pt_index].z;

  // TODO: switch to lrintf for speed maybe
  long x0 = lroundf(cx - sigma);
  long x1 = lroundf(cx + sigma + 1);
  long y0 = lroundf(cy - sigma);
  long y1 = lroundf(cy + sigma + 1);

  long v = (long)integral_img(y1, x1)
         + integral_img(y0, x0)
         - integral_img(y1, x0)
         - integral_img(y0, x1);

  return floorf(v / ((x1 - x0)*(y1 - y0)));
}

__global__ void describeKeypoints(
    FreakGpu::FreakConsts consts,
    const cv::cudev::GlobPtr<uint> integral_img,
    const CudaDeviceVector<short3>::Dev keypoints,
    cv::cuda::PtrStepSzb descriptors) {

  __shared__ float all_points[kKeyPointsPerBlock][FreakGpu::kPoints];
  __shared__ float2 all_orientation_weights[kKeyPointsPerBlock][kNumThreads];

  float* points = all_points[threadIdx.y];
  float2* orientation_weight = all_orientation_weights[threadIdx.y];

  int tid = threadIdx.x;

  int kp_id = blockDim.y * blockIdx.x + threadIdx.y;
  bool active = kp_id < keypoints.size();

  // Compute points
  if (tid < FreakGpu::kPoints) {
    points[tid] = 
        active ? computePoint(consts, integral_img, keypoints[kp_id], tid) : 0;
  }

  __syncthreads();

  float2 w = make_float2(0.0f, 0.0f);

  for (int i = tid; i < FreakGpu::kOrientationPairs; i += kNumThreads) {
    const short4& p = consts.orientation_pairs[i];
    float d = points[p.x] - points[p.y];      
    w.x += bad_floorf(d * p.z / 2048);
    w.y += bad_floorf(d * p.w / 2048);
  }

  /* __syncthreads(); */

  orientation_weight[tid] = w;

  __syncthreads();

  float2 dv = reduce(orientation_weight); 


  float angle = active ? atan2f(dv.y, dv.x) * 180.0/CV_PI : 0;
  long orientation = (bad_floorf(FreakGpu::kOrientations * angle / 360.0 + 0.5));
  if (orientation < 0) {
    orientation += FreakGpu::kOrientations;
  }
  if (orientation >= FreakGpu::kOrientations) {
    orientation -= FreakGpu::kOrientations;
  }


  // Compute points for the orientation
  if (tid < FreakGpu::kPoints) {
    points[tid] = active 
      ? computePoint(
          consts, integral_img, 
          keypoints[kp_id], 
          tid + orientation * FreakGpu::kPoints) 
      : 0;
  }

  __syncthreads();

  if (active) {
    // Compute the descriptor
    uint desc_byte = 0;
    int b = 0;
    
    for (int i = tid; i < FreakGpu::kPairs; i += kNumThreads, b++) {
      short2 p = consts.descriptor_pairs[i];
      int v = points[p.x] >= points[p.y];

      desc_byte |= (v << b);
    }
  
    descriptors(kp_id, tid) = desc_byte;
  }
}

template <class T>
void uploadVector(const std::vector<T>& v, T** ptr) {
  int size = sizeof(T)*v.size();
  cudaSafeCall(cudaMalloc(ptr, size));
  cudaSafeCall(cudaMemcpy(*ptr, v.data(), size, cudaMemcpyHostToDevice));
};

FreakGpu::FreakGpu(double feature_size) : FreakBase(feature_size) {
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

  uploadVector(points, &consts_.points);
  uploadVector(orientation_pairs, &consts_.orientation_pairs);
  uploadVector(descriptor_pairs, &consts_.descriptor_pairs);
}

FreakGpu::~FreakGpu() {
  cudaSafeCall(cudaFree(consts_.points));
  cudaSafeCall(cudaFree(consts_.orientation_pairs));
  cudaSafeCall(cudaFree(consts_.descriptor_pairs));
}

void FreakGpu::describe(
    const cv::cuda::GpuMat& img,
    const CudaDeviceVector<short3>& keypoints,
    int keypoints_count,
    cv::cudev::GpuMat_<uint8_t>& descriptors,
    cv::cuda::Stream& stream) {

  cv::cuda::integral(img, integral_, stream);

  dim3 thread_block_dim(kNumThreads, kKeyPointsPerBlock);
  dim3 grid_dim(
      (keypoints_count + kKeyPointsPerBlock - 1)/ kKeyPointsPerBlock);

  auto cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

  describeKeypoints<<<grid_dim, thread_block_dim, 0, cuda_stream>>>(
    consts_,
    integral_, 
    keypoints,
    descriptors);

  cudaSafeCall(cudaGetLastError());
}


