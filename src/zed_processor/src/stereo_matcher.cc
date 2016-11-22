#include "stereo_matcher.hpp"

#include <opencv2/cudev/ptr2d/gpumat.hpp>

namespace stereo_matcher {
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
      ushort2* m);

  cv::cudev::GpuMat_<cv::Vec2s> match(
      const cv::cudev::GpuMat_<uchar>& d1,
      const cv::cudev::GpuMat_<uchar>& d2,
      const cv::cudev::GpuMat_<cv::Vec2s>& pairs,
      float threshold_ratio,
      cv::cudev::GpuMat_<cv::Vec4s>& m1,
      cv::cudev::GpuMat_<cv::Vec4s>& m2,
      cv::cudev::GpuMat_<cv::Vec2s>& m) {
    assert(d1.cols == 64);
    assert(d2.cols == 64);
    assert(m1.cols == d1.rows);
    assert(m2.cols == d2.rows);
    assert(m.cols == m1.cols);

    int cnt = gpu_match(
        d1, d2, 
        pairs.ptr<ushort2>(), pairs.cols, 
        threshold_ratio, 
        m1.ptr<ushort4>(), m1.cols, 
        m2.ptr<ushort4>(), m2.cols, 
        m.ptr<ushort2>());

    return m.colRange(0, cnt);
  }
}
