#ifndef __FAST_GPU__HPP__
#define __FAST_GPU__HPP__

#include <opencv2/core.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

class FastGpu {
  public:
    FastGpu(int max_keypoints);

    void detect(const cv::cudev::GpuMat_<uchar>& img, int threshold);

    const cv::cudev::GpuMat_<uint8_t>& scores() const { return scores_; }
    cv::cudev::GpuMat_<cv::Vec3s> keypoints() const { 
      return final_keypoints_.colRange(0, keypoint_count_); 
    }

  private:
    cv::cudev::GpuMat_<uint8_t> scores_;

    cv::cudev::GpuMat_<cv::Vec2s> tmp_keypoints_;
    cv::cudev::GpuMat_<cv::Vec3s> final_keypoints_;

    int keypoint_count_;
};

#endif

