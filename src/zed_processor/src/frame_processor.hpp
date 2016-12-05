#ifndef __FRAME_PROCESSOR__HPP__
#define __FRAME_PROCESSOR__HPP__

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

#include "cuda_device_vector.hpp"
#include "cuda_pinned_allocator.hpp"
#include "fast_gpu.hpp"
#include "freak_gpu.hpp"
#include "stereo_matcher.hpp"

struct StereoCalibrationData;

const int kMaxKeypoints = 20000;

struct StereoPoint {
  // World coordinates
  cv::Point3f world;
  // Keypoints (x, y, response)
  cv::Point2f left, right;
  int left_i, right_i;
  int score;
};

struct FrameData {
  FrameData(int max_points) {
    points.reserve(max_points);
    descriptors_left.create(max_points, FreakGpu::kDescriptorWidth);
    descriptors_right.create(max_points, FreakGpu::kDescriptorWidth);
    d_left.create(max_points, FreakGpu::kDescriptorWidth);
    d_right.create(max_points, FreakGpu::kDescriptorWidth);
  }

  std::vector<StereoPoint> points;
  cv::Mat_<uint8_t> descriptors_left, descriptors_right;
  cv::cudev::GpuMat_<uint8_t> d_left, d_right;
};

class FrameProcessor {
  public:
    FrameProcessor(const StereoCalibrationData& calib);
    ~FrameProcessor();
    
    FrameProcessor(const FrameProcessor&) = delete;
    FrameProcessor& operator=(const FrameProcessor&) = delete;
    
    void process(
        const cv::Mat src[], 
        int threshold,
        FrameData& frame_data);

  private:
    static void computeKpPairs_(
        const PinnedVector<short3>& kps1,
        const PinnedVector<short3>& kps2,
        PinnedVector<ushort2>& keypoint_pairs);

  private:
    const StereoCalibrationData* calib_;

    FreakGpu freak_;
    FastGpu fast_l_, fast_r_;

    Matcher matcher_;
    
    cv::cuda::Stream streams_[3];
    cv::cuda::Event events_[2];

    cv::cuda::GpuMat undistort_map_x_[2], undistort_map_y_[2];
    cv::cudev::GpuMat_<uchar> src_img_[2], undistorted_image_gpu_[2];
    cv::cudev::GpuMat_<uint> integral_image_gpu_[2];

    CudaDeviceVector<short3> keypoints_gpu_l_, keypoints_gpu_r_;

    // [N]: a list of keypoints detected in the left and right image
    // each keypoint is represented as x, y, reponse
    PinnedVector<short3> keypoints_cpu_[2];
    int* keypoint_sizes_;
    // [NxD]: descriptors corresponding to the keypoints_
    cv::cudev::GpuMat_<uint8_t> descriptors_gpu_[2];
    // Descriptor pair candidates to match
    PinnedVector<ushort2> keypoint_pairs_;
    CudaDeviceVector<ushort2> keypoint_pairs_gpu_;
    // Matches
    std::vector<ushort2> matches_;
    CudaDeviceVector<ushort2> matches_gpu_;
};

#endif
