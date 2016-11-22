#include <stdio.h>

#include <cuda_runtime.h>

#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/glob.hpp>

namespace freak_gpu {
  const int kNumThreads = 64;

  const int kNumPoints = 43;
  const int kNumOrientations = 256;
  const int kNumOrientationPairs = 45;
  const int kNumDescriptorPairs = 512;

  const int kKeyPointsPerBlock = 2;

  __device__ float3 kPoints[kNumPoints * kNumOrientations];
  __device__ short4 kOrientationPairs[kNumOrientationPairs];
  __device__ short2 kDescriptorPairs[kNumDescriptorPairs];

  __device__ float computePoint(
      const cv::cudev::GlobPtr<uint> integral_img,
      short3 center,
      int pt_index) {
    float cx = center.x + kPoints[pt_index].x;
    float cy = center.y + kPoints[pt_index].y;
    float sigma = kPoints[pt_index].z;

    long x0 = lrintf(cx - sigma);
    long x1 = lrintf(cx + sigma + 1);
    long y0 = lrintf(cy - sigma);
    long y1 = lrintf(cy + sigma + 1);

    long v = (long)integral_img(y1, x1)
           + integral_img(y0, x0)
           - integral_img(y1, x0)
           - integral_img(y0, x1);

    /* printf( */
    /*     "Point (%d, %d) %d -> (%ld, %ld) - (%ld, %ld) = %ld\n", */ 
    /*     center.x, center.y, pt_index, */
    /*     x0, y0, x1, y1, v); */

    return floorf(v / ((x1 - x0)*(y1 - y0)));
  }

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

  __global__ void describeKeypoints(
      const cv::cudev::GlobPtr<uint> integral_img,
      const short3* keypoints,
      int keypoint_count,
      cv::cuda::PtrStepSzb descriptors) {

    __shared__ float all_points[kKeyPointsPerBlock][kNumPoints];
    __shared__ float2 all_orientation_weights[kKeyPointsPerBlock][kNumThreads];
  
    float* points = all_points[threadIdx.y];
    float2* orientation_weight = all_orientation_weights[threadIdx.y];

    int tid = threadIdx.x;

    int kp_id = blockDim.y * blockIdx.x + threadIdx.y;
    bool active = kp_id < keypoint_count;

    // Compute points
    if (tid < kNumPoints) {
      points[tid] = 
        active ? computePoint(integral_img, keypoints[kp_id], tid) : 0;
    }

    /* __syncthreads(); */
 
    /* if (tid == 0 && active) { */
    /*   printf("Points:"); */
    /*   for (int i=0; i < kNumPoints; ++i) { */
    /*     printf(" %ld", (long)points[i]); */
    /*   } */
    /*   printf("\n"); */
    /* } */

    __syncthreads();


    float2 w = make_float2(0.0f, 0.0f);

    for (int i = tid; i < kNumOrientationPairs; i += kNumThreads) {
      const short4& p = kOrientationPairs[i];
      float d = points[p.x] - points[p.y];      
      w.x += bad_floorf(d * p.z / 2048);
      w.y += bad_floorf(d * p.w / 2048);

      /* if (active) { */
      /*   printf("o %d: %d %d %d %d %f\n", i, p.x, p.y, p.z, p.w, bad_floorf(d * p.z / 2048)); */
      /* } */
    }

    /* __syncthreads(); */

    /* if (active) { */
    /*   printf("%d: %f %f\n", tid, w.x, w.y); */
    /* } */

    orientation_weight[tid] = w;

    __syncthreads();

    float2 dv = reduce(orientation_weight); 

    /* if (tid == 0 && active) { */
    /*   printf("GPU dv: (%f, %f)\n", dv.x, dv.y); */
    /* } */

    float angle = active ? atan2f(dv.y, dv.x) * 180.0/CV_PI : 0;
    long orientation = (bad_floorf(kNumOrientations * angle / 360.0 + 0.5));
    if (orientation < 0) {
      orientation += kNumOrientations;
    }
    if (orientation >= kNumOrientations) {
      orientation -= kNumOrientations;
    }

    /* if (tid == 0 && active) { */
    /*   printf("GPU orientation: %f %d\n", kNumOrientations * angle / 360.0, orientation); */
    /* } */

    // Compute points for the orientation
    if (tid < kNumPoints) {
      points[tid] = active 
        ? computePoint(
            integral_img, keypoints[kp_id], tid + orientation * kNumPoints) 
        : 0;
    }

    __syncthreads();
 
    /* if (tid == 0 && active) { */
    /*   printf("GPU Points:"); */
    /*   for (int i=0; i < kNumPoints; ++i) { */
    /*     printf(" %ld", (long)points[i]); */
    /*   } */
    /*   printf("\n"); */
    /* } */
    /* __syncthreads(); */

    if (active) {
      // Compute the descriptor
      uint desc_byte = 0;
      int b = 0;
      
      for (int i = tid; i < kNumDescriptorPairs; i += kNumThreads, b++) {
        short2 p = kDescriptorPairs[i];
        int v = points[p.x] >= points[p.y];

        desc_byte |= (v << b);
      }
    
      descriptors(kp_id, tid) = desc_byte;
    }
  }

  __host__ void describeKeypointsGpu(
      const cv::cudev::GlobPtr<uint> integral_img,
      const short3* keypoints,
      int keypoint_count,
      cv::cuda::PtrStepSzb descriptors) {

    dim3 thread_block_dim(kNumThreads, kKeyPointsPerBlock);
    dim3 grid_dim(
        (keypoint_count + kKeyPointsPerBlock - 1)/ kKeyPointsPerBlock);

    describeKeypoints<<<grid_dim, thread_block_dim>>>(
        integral_img,
        keypoints,
        keypoint_count,
        descriptors);

    cudaSafeCall(cudaGetLastError());
  }

  __host__ bool initialize(
      float3* points, int num_points,
      short4* orientation_pairs, int num_orientation_pairs,
      short2* descriptor_pairs, int num_descriptor_pairs) {

    if (num_points != kNumPoints * kNumOrientations) {
      printf("Num points mismatch: %d %d\n", num_points, kNumPoints);
      return false;
    }

    void* points_dev;
    cudaSafeCall(cudaGetSymbolAddress(&points_dev, kPoints));
    cudaSafeCall(cudaMemcpy(
        points_dev,
        points, sizeof(float3) * num_points, 
        cudaMemcpyDefault));

    if (num_orientation_pairs != kNumOrientationPairs) {
      printf("Num orientation pairs mismatch: %d %d\n", 
          num_orientation_pairs, kNumOrientationPairs);
      return false;
    }

    void* orientation_pairs_dev;
    cudaSafeCall(cudaGetSymbolAddress(
          &orientation_pairs_dev, kOrientationPairs));
    cudaSafeCall(cudaMemcpy(
        orientation_pairs_dev,
        orientation_pairs, sizeof(short4) * num_orientation_pairs, 
        cudaMemcpyDefault));

    if (num_descriptor_pairs != kNumDescriptorPairs) {
      printf("Num descriptor pairs mismatch: %d %d\n", 
          num_descriptor_pairs, kNumDescriptorPairs);
      return false;
    }

    void* descriptor_pairs_dev;
    cudaSafeCall(cudaGetSymbolAddress(&descriptor_pairs_dev, kDescriptorPairs));
    cudaSafeCall(cudaMemcpy(
        descriptor_pairs_dev,
        descriptor_pairs, sizeof(short2) * num_descriptor_pairs, 
        cudaMemcpyDefault));

    return true;
  }
}
