#ifndef __FAST_GPU__HPP__
#define __FAST_GPU__HPP__

#include <opencv2/core.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#include "cuda_device_vector.hpp"

class FastGpu {
  public:
    FastGpu(int max_unsuppressed_keypoints, int border);
    ~FastGpu();

    void computeScores(
        const cv::cudev::GpuMat_<uchar>& img, 
        int threshold,
        cv::cuda::Stream& s);

    void downloadKpCount(cv::cuda::Stream& s);

    void extract(
        int threshold, 
        CudaDeviceVector<short3>& res, 
        cv::cuda::Stream& s);


    // Helper, blocking
    void detect(
        const cv::cudev::GpuMat_<uchar>& img, 
        int threshold,
        CudaDeviceVector<short3>& res);
  private:
    int border_;
    int* n_tmp_keypoints_;

    cv::cudev::GpuMat_<uint8_t> scores_;
    CudaDeviceVector<short2> tmp_keypoints_;
};

#endif

