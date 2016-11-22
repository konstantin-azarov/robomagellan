#include <cuda_runtime.h>

#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/glob.hpp>

namespace stereo_matcher {
  __device__ int match_count;

  const int kPairsPerBlock = 16;

  __global__ void match(
      const cv::cudev::GlobPtr<uchar> d1,
      const cv::cudev::GlobPtr<uchar> d2,
      const ushort2* pairs,
      int n_pairs,
      ushort4* m1,
      ushort4* m2) {
   
    __shared__ uint all_scores[kPairsPerBlock][32];
    uint* scores = all_scores[threadIdx.y];

    uint pair_id = kPairsPerBlock * blockIdx.x + threadIdx.y;
    uint tid = threadIdx.x;

    ushort2 p = pairs[pair_id];

    if (pair_id < n_pairs) {
      uint tid = threadIdx.x;

      ushort2 p = pairs[pair_id];

      ushort b1 = *(reinterpret_cast<const ushort*>(d1.row(p.x)) + tid);
      ushort b2 = *(reinterpret_cast<const ushort*>(d2.row(p.y)) + tid);

      scores[tid] = __popc(b1 ^ b2);
   }

    __syncthreads();

    if (pair_id < n_pairs) {
      #pragma unroll
      for (uint s = 32/2; s > 0; s /= 2) {
        if (tid < s) {
          scores[tid] += scores[tid + s];
        }
        
        __syncthreads();
      }
    }

    uint score = scores[0];
   
    /* if (tid == 0 && pair_id < n_pairs) { */
    /*   printf("P%d, %d: %d %d %d\n", pair_id, tid, p.x, p.y, score); */
    /* } */

    return;

    if (pair_id < n_pairs && tid <= 1) {
      ushort i = tid == 0 ? p.x : p.y;
      ushort j = tid == 0 ? p.y : p.x;
      unsigned long long* m = 
        reinterpret_cast<unsigned long long*>(tid == 0 ? m1 : m2) + i;


      bool ok = false;
      do {
        unsigned long long old_v = *m;
        unsigned long long new_v;

        const ushort4& old_entry = *reinterpret_cast<const ushort4*>(&old_v);
        ushort4& new_entry = *reinterpret_cast<ushort4*>(&new_v);
        bool updated = false;
        if (score < old_entry.x) {
          new_entry = make_ushort4(score, old_entry.x, j, 0);
          updated = true;
        } else if (score < old_entry.y) {
          new_entry = make_ushort4(old_entry.x, score, old_entry.z, 0);
          updated = true;
        }

//        ok = updated ? atomicCAS(m, old_v, new_v) == old_v : true;
        *m = new_v;
        ok = true;
      } while (!ok);
    }
  }

  int gpu_match(
      const cv::cudev::GlobPtr<uchar> d1,
      const cv::cudev::GlobPtr<uchar> d2,
      const ushort2* pairs,
      int n_pairs,
      float threshold_ratio,
      ushort4* m1,
      int n1,
      ushort4* m2,
      int n2,
      ushort2* m) {
    
    int* match_count_dev;
    cudaSafeCall(cudaGetSymbolAddress((void**)&match_count_dev, match_count));
    cudaSafeCall(cudaMemset(match_count_dev, 0, sizeof(int)));

    cudaSafeCall(cudaMemset(m1, 0xff, n1*sizeof(ushort4)));
    cudaSafeCall(cudaMemset(m2, 0xff, n2*sizeof(ushort4)));

    int n_blocks = (n_pairs + kPairsPerBlock - 1) / kPairsPerBlock;

    match<<<n_blocks, dim3(32, kPairsPerBlock)>>>(
        d1, d2, pairs, n_pairs, m1, m2);

    cudaSafeCall(cudaGetLastError());

    return 0;
  }
  
}
