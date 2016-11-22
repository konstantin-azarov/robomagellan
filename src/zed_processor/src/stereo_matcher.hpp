#ifndef __STEREO_MATCHER__HPP__
#define __STEREO_MATCHER__HPP__

#include <opencv2/cudev/ptr2d/gpumat.hpp>

namespace stereo_matcher {
  /** 
   * m1, m2 and m should be of an appropriate size
   */
  cv::cudev::GpuMat_<cv::Vec2s> match(
      const cv::cudev::GpuMat_<uchar>& d1,
      const cv::cudev::GpuMat_<uchar>& d2,
      const cv::cudev::GpuMat_<cv::Vec2s>& pairs,
      float threshold_ratio,
      cv::cudev::GpuMat_<cv::Vec4w>& m1,
      cv::cudev::GpuMat_<cv::Vec4w>& m2,
      cv::cudev::GpuMat_<cv::Vec2s>& m);

}

#endif

