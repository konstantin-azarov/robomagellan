#include "fast_gpu.hpp"

namespace fast_gpu {
  extern int detect(
      const cv::cudev::GlobPtr<uchar> img,
      short2 img_size,
      int threshold,
      int border,
      cv::cudev::GlobPtr<uchar> scores, 
      short2* tmp_keypoints_dev,
      short3* keypoints_dev,
      int max_keypoints);
};

FastGpu::FastGpu(int max_keypoints, int border) : border_(border) {
  tmp_keypoints_.create(1, max_keypoints);
  final_keypoints_.create(1, max_keypoints);
}

void FastGpu::detect(const cv::cudev::GpuMat_<uchar>& img, int threshold) {
  scores_.create(img.rows, img.cols);
  scores_.setTo(0);

  keypoint_count_ = fast_gpu::detect(
      img,
      make_short2(img.cols, img.rows),
      threshold,
      border_,
      scores_,
      tmp_keypoints_.ptr<short2>(),
      final_keypoints_.ptr<short3>(),
      tmp_keypoints_.cols);
}


