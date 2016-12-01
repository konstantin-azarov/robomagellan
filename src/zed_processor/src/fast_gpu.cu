#include <stdio.h>

#include <cuda_runtime.h>

#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/glob.hpp>

#include "fast_gpu.hpp"

namespace {
  const int BX = 16;
  const int BY = 16;

  __global__ void compute_scores(
      const cv::cudev::GlobPtr<uchar> img,
      short2 img_size,
      int threshold, int border,
      cv::cudev::GlobPtr<uchar> scores,
      CudaDeviceVector<short2>::Dev keypoints) {

    __shared__ uchar tile[BY + 6][BX + 6];
    int strip[16];

    int px = threadIdx.x + blockDim.x * blockIdx.x + border;
    int py = threadIdx.y + blockDim.y * blockIdx.y + border;

    int lx = px - 3;
    int ly = py - 3;

    // Copy data
    if (ly < img_size.y && lx < img_size.x) {
      tile[threadIdx.y][threadIdx.x] = img(ly, lx);
    }

    if (ly < img_size.y && ly + BX < img_size.x && threadIdx.x < 6) {
      tile[threadIdx.y][threadIdx.x + BX] = img(ly, lx + BX);
    }

    if (lx < img_size.x && ly + BY < img_size.y && threadIdx.y < 6) {
      tile[threadIdx.y + BY][threadIdx.x] = img(ly + BY, lx);
    }

    if (threadIdx.x >= 10 && threadIdx.y >= 10 &&
        lx + 6 < img_size.x && ly + 6 < img_size.y) {
      tile[threadIdx.y + 6][threadIdx.x + 6] = img(ly + 6, lx + 6);
    }

    __syncthreads();

    // Copy strip

    int x = threadIdx.x + 3;
    int y = threadIdx.y + 3;

    int score = 0;

    if (px < img_size.x - border && py < img_size.y - border) {
      int v0 = tile[y][x];

      strip[0] = tile[y - 3][x - 1] - v0;
      strip[1] = tile[y - 3][x] - v0;
      strip[2] = tile[y - 3][x + 1] - v0;
      strip[3] = tile[y - 2][x + 2] - v0;
      strip[4] = tile[y - 1][x + 3] - v0;
      strip[5] = tile[y    ][x + 3] - v0;
      strip[6] = tile[y + 1][x + 3] - v0;
      strip[7] = tile[y + 2][x + 2] - v0;
      strip[8] = tile[y + 3][x + 1] - v0;
      strip[9] = tile[y + 3][x] - v0;
      strip[10] = tile[y + 3][x - 1] - v0;
      strip[11] = tile[y + 2][x - 2] - v0;
      strip[12] = tile[y + 1][x - 3] - v0;
      strip[13] = tile[y    ][x - 3] - v0;
      strip[14] = tile[y - 1][x - 3] - v0;
      strip[15] = tile[y - 2][x - 2] - v0;
   
      for (int i0=0; i0 < 16; i0++) {
        int sp = max(strip[i0], strip[(i0 + 1) % 16]);
        sp = max(sp, strip[(i0 + 2) % 16]);
        if (sp >= -threshold) {
          continue;
        }

        sp = max(sp, strip[(i0 + 3) % 16]);
        sp = max(sp, strip[(i0 + 4) % 16]);
        sp = max(sp, strip[(i0 + 5) % 16]);
        sp = max(sp, strip[(i0 + 6) % 16]);
        sp = max(sp, strip[(i0 + 7) % 16]);
        sp = max(sp, strip[(i0 + 8) % 16]);

        score = min(score, sp);
      }

      score = -score;
   
      // Compute score
      for (int i0=0; i0 < 16; i0++) {
        int sp = min(strip[i0], strip[(i0 + 1) % 16]);
        sp = min(sp, strip[(i0 + 2) % 16]);
        if (sp <= threshold) {
          continue;
        }

        sp = min(sp, strip[(i0 + 3) % 16]);
        sp = min(sp, strip[(i0 + 4) % 16]);
        sp = min(sp, strip[(i0 + 5) % 16]);
        sp = min(sp, strip[(i0 + 6) % 16]);
        sp = min(sp, strip[(i0 + 7) % 16]);
        sp = min(sp, strip[(i0 + 8) % 16]);

        score = max(score, sp);
      }
    }

    __syncthreads();

    tile[threadIdx.y][threadIdx.x] = score;

    __syncthreads();

    bool good = true;
    // Nonmax supression
    if (threadIdx.x > 0 && threadIdx.y > 0 &&
        threadIdx.x < BX - 1 && threadIdx.y < BY - 1) {
      int x = threadIdx.x;
      int y = threadIdx.y;

      good = 
        tile[y-1][x-1] < score &&
        tile[y-1][x  ] < score &&
        tile[y-1][x+1] < score &&
        tile[y  ][x-1] < score &&
        tile[y  ][x+1] < score &&
        tile[y+1][x-1] < score &&
        tile[y+1][x  ] < score &&
        tile[y+1][x+1] < score;
    }

    // Copy scores back
    if (score > threshold) {
      scores(py, px) = score;

      if (good) {
        keypoints.push(make_short2(px, py));
      }
    }
  }

  __global__ void nonmax_supression(
      cv::cudev::GlobPtr<uchar> scores,
      const CudaDeviceVector<short2>::Dev keypoints,
      CudaDeviceVector<short3>::Dev res) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < keypoints.size()) {
      short2 kp = keypoints[i];
      int score = scores(kp.y, kp.x);

      bool good = 
        scores(kp.y-1, kp.x-1) < score &&
        scores(kp.y-1, kp.x  ) < score &&
        scores(kp.y-1, kp.x+1) < score &&
        scores(kp.y  , kp.x-1) < score &&
        scores(kp.y  , kp.x+1) < score &&
        scores(kp.y+1, kp.x-1) < score &&
        scores(kp.y+1, kp.x  ) < score &&
        scores(kp.y+1, kp.x+1) < score;
      if (good) {
        res.push(make_short3(kp.x, kp.y, score - 1));
      }
    }
  }
}

FastGpu::FastGpu(int max_keypoints, int border) : 
  border_(border),
  tmp_keypoints_(max_keypoints) {
  cudaHostAlloc(&n_tmp_keypoints_, sizeof(int), cudaHostAllocDefault);
}

FastGpu::~FastGpu() {
  cudaFreeHost(n_tmp_keypoints_);
}

#include <chrono>
#include <iostream>

using namespace std::chrono;

void FastGpu::computeScores(
    const cv::cudev::GpuMat_<uchar>& img, 
    int threshold,
    cv::cuda::Stream& s) {
  /* auto t0 = std::chrono::high_resolution_clock::now(); */

  scores_.create(img.rows, img.cols);
  scores_.setTo(0, s);
  
  /* auto t1 = std::chrono::high_resolution_clock::now(); */

  dim3 grid_dim((img.cols + BX - 1) / BX, (img.rows + BY - 1) / BY);
  dim3 thread_block_dim(BX, BY);

  tmp_keypoints_.clear(s);

  /* auto t2 = std::chrono::high_resolution_clock::now(); */

  auto cuda_stream = cv::cuda::StreamAccessor::getStream(s);
  compute_scores<<<grid_dim, thread_block_dim, 0, cuda_stream>>>(
      img, 
      make_short2(img.cols, img.rows),
      threshold, 
      border_,
      scores_, 
      tmp_keypoints_);
}

void FastGpu::downloadKpCount(cv::cuda::Stream& s) {
  tmp_keypoints_.sizeAsync(*n_tmp_keypoints_, s);
}

void FastGpu::extract(
    int threshold,
    CudaDeviceVector<short3>& res,
    cv::cuda::Stream& s) {
  res.clear(s);
  auto cuda_stream = cv::cuda::StreamAccessor::getStream(s);
  nonmax_supression<<<(*n_tmp_keypoints_ + 63)/64, 64, 0, cuda_stream>>>(
      scores_, tmp_keypoints_, res);
}

void FastGpu::detect(
    const cv::cudev::GpuMat_<uchar>& img, 
    int threshold,
    CudaDeviceVector<short3>& res) {
  auto s = cv::cuda::Stream::Null();

  computeScores(img, threshold, s);
  downloadKpCount(s);
  s.waitForCompletion();
  extract(threshold, res, s);
}
