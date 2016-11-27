#ifndef __FRAME_PROCESSOR__HPP__
#define __FRAME_PROCESSOR__HPP__

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

#include "cuda_device_vector.hpp"
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
  }

  std::vector<StereoPoint> points;
  cv::Mat_<uint8_t> descriptors_left, descriptors_right;
};

class FrameProcessor {
  public:
    FrameProcessor(const StereoCalibrationData& calib);
    
    FrameProcessor(const FrameProcessor&) = delete;
    FrameProcessor& operator=(const FrameProcessor&) = delete;
    
    void process(
        const cv::Mat src[], 
        int threshold,
        FrameData& frame_data);

    /* const cv::Mat& undistortedImage(int i) const { */ 
    /*   return undistorted_image_[i]; */
    /* } */

    /* const std::vector<cv::Point3d>& points() const { */
    /*   return points_; */
    /* } */

    /* // For 0 <= p < points_.size() return left and right features for the */ 
    /* // corresponding point. */
    /* const std::pair<cv::Point2f, cv::Point2f> features(int p) const { */
    /*   int i = point_keypoints_[p]; */
    /*   int j = matches_[0][i].z; */

    /*   cv::Point2f a(keypoints_cpu_[0][i].x, keypoints_cpu_[0][i].y); */
    /*   cv::Point2f b(keypoints_cpu_[1][j].x, keypoints_cpu_[1][j].y); */
    /*   return std::make_pair(a, b); */
    /* } */


    /* // For 0 <= p < points_.size() return left and right descriptor for the */
    /* // corresponding point. */
    /* const std::pair<cv::Mat, cv::Mat> pointDescriptors(int p) const { */
    /*   int i = point_keypoints_[p]; */
    /*   int j = matches_[0][i].z; */

    /*   return std::make_pair(descriptors_[0].row(i), descriptors_[1].row(j)); */
    /* } */

    /* const std::vector<int>& pointKeypoints() const { return point_keypoints_; } */
    /* const std::vector<short3>& keypoints(int t) const  { return keypoints_cpu_[t]; } */
    /* cv::Mat descriptors(int t) const { return descriptors_[t]; } */
    /* std::vector<int> matches(int t) const { */ 
    /*   std::vector<int> res; */
    /*   for (const auto& m : matches_[t]) { */
    /*     res.push_back(m.z == 0xFFFF ? -1 : m.z); */
    /*   } */
    /*   return res; */
    /* } */
    
  private:
    static void computeKpPairs_(
        const std::vector<short3>& kps1,
        const std::vector<short3>& kps2,
        std::vector<ushort2>& keypoint_pairs);

    /**
     * d1 - left descriptors
     * d2 - right descriptors
     * keypoint_pairs - potential matches (left_index, right_index)
     * matches[t][i] = (best_score, second_best_score, best_index)
     */
    static void computeMatches_(
        const cv::Mat_<uchar>& d1,
        const cv::Mat_<uchar>& d2,
        const std::vector<short2>& keypoint_pairs,
        std::vector<ushort4> matches[2]);

    void match(const std::vector<short3>& kps1,
               const cv::Mat& desc1,
               const std::vector<short3>& kps2,
               const cv::Mat& desc2,
               int inv,
               std::vector<int>& matches);

  private:
    const StereoCalibrationData* calib_;

    FreakGpu freak_;
    FastGpu fast_;

    Matcher matcher_;

    cv::cuda::GpuMat undistort_map_x_[2], undistort_map_y_[2];
    cv::cudev::GpuMat_<uchar> src_img_[2], undistorted_image_gpu_[2];

    cv::Mat undistorted_image_[2];

    CudaDeviceVector<short3> keypoints_gpu_l_, keypoints_gpu_r_;

    // [N]: a list of keypoints detected in the left and right image
    // each keypoint is represented as x, y, reponse
    std::vector<short3> keypoints_cpu_[2];
    // [NxD]: descriptors corresponding to the keypoints_
    cv::cudev::GpuMat_<uint8_t> descriptors_gpu_[2];
    // Descriptor pair candidates to match
    std::vector<ushort2> keypoint_pairs_;
    CudaDeviceVector<ushort2> keypoint_pairs_gpu_;
    // Matches
    std::vector<cv::Vec2s> matches_;
};

#endif
