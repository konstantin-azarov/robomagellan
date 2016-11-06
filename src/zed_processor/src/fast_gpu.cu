#include <stdio.h>

#include <cuda_runtime.h>

#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/glob.hpp>

namespace fast_gpu {
  const int BX = 16;
  const int BY = 16;

  __global__ void compute_scores(
      const cv::cudev::GlobPtr<uchar> img,
      short2 img_size,
      int threshold, int border,
      cv::cudev::GlobPtr<uchar> scores,
      short2* keypoints,
      int* kp_index,
      int max_keypoints) {

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
        int idx = atomicAdd(kp_index, 1);
        if (idx < max_keypoints) {
          keypoints[idx] = make_short2(px, py);
        }
      }
    }
  }

  __global__ void nonmax_supression(
      cv::cudev::GlobPtr<uchar> scores,
      short2* keypoints,
      int count,
      short3* final_keypoints,
      int* kp_index,
      int max_keypoints) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < count) {
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
        int idx = atomicAdd(kp_index, 1);
        if (idx < max_keypoints) {
          final_keypoints[idx] = make_short3(kp.x, kp.y, score);
        }
      }
    }
  }
  
  __device__ int kp_index;

  __host__ int detect(
      const cv::cudev::GlobPtr<uchar> img,
      short2 img_size,
      int threshold,
      cv::cudev::GlobPtr<uchar> scores, 
      short2* tmp_keypoints_dev,
      short3* keypoints_dev,
      int max_keypoints) {
    dim3 grid_dim((img_size.x + BX - 1) / BX, (img_size.y + BY - 1) / BY);
    dim3 thread_block_dim(BX, BY);

    int* kp_index_dev;
    cudaSafeCall(cudaGetSymbolAddress((void**)&kp_index_dev, kp_index));
    cudaSafeCall(cudaMemset(kp_index_dev, 0, sizeof(int)));
  
    compute_scores<<<grid_dim, thread_block_dim>>>(
        img, 
        img_size, 
        threshold, 
        3,
        scores, 
        tmp_keypoints_dev, 
        kp_index_dev, 
        max_keypoints);

    int n_keypoints;
    cudaSafeCall(cudaMemcpy(
        &n_keypoints, kp_index_dev, sizeof(int), cudaMemcpyDeviceToHost)); 
    cudaSafeCall(cudaMemset(kp_index_dev, 0, sizeof(int)));

    nonmax_supression<<<(n_keypoints + 63)/64, 64>>>(
        scores,
        tmp_keypoints_dev,
        n_keypoints,
        keypoints_dev,
        kp_index_dev,
        max_keypoints);


    cudaSafeCall(cudaMemcpy(
        &n_keypoints, kp_index_dev, sizeof(int), cudaMemcpyDeviceToHost)); 
    
    cudaSafeCall(cudaDeviceSynchronize());

    return n_keypoints;
  }
}
