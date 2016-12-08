#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <nvToolsExtCuda.h>

#include "calibration_data.hpp"
#include "descriptor_tools.hpp"
#include "frame_processor.hpp"
#include "math3d.hpp"

using namespace std::chrono;


FrameProcessor::FrameProcessor(
    const StereoCalibrationData& calib,
    const FrameProcessorConfig& config) : 
    calib_(&calib),
    freak_(config.descriptor_radius),
    fast_l_(kMaxKeypoints*5, freak_.borderWidth()),
    fast_r_(kMaxKeypoints*5, freak_.borderWidth()),
    matcher_(kMaxKeypoints, kMaxKeypoints * 15),
    matches_gpu_(kMaxKeypoints),
    keypoints_gpu_l_(kMaxKeypoints),
    keypoints_gpu_r_(kMaxKeypoints),
    keypoint_pairs_gpu_(kMaxKeypoints * 15) {

  keypoints_cpu_[0].reserve(kMaxKeypoints);
  keypoints_cpu_[1].reserve(kMaxKeypoints);

  descriptors_gpu_[0].create(kMaxKeypoints, FreakGpu::kDescriptorWidth);
  descriptors_gpu_[1].create(kMaxKeypoints, FreakGpu::kDescriptorWidth);

  keypoint_pairs_.reserve(kMaxKeypoints * 15);

  for (int i=0; i < 2; ++i) {
    undistort_map_x_[i].upload(calib_->undistort_maps[i].x);
    undistort_map_y_[i].upload(calib_->undistort_maps[i].y);
  }

  matches_.reserve(kMaxKeypoints);

  cudaSafeCall(cudaHostAlloc(
        &keypoint_sizes_, 2*sizeof(int), cudaHostAllocDefault));
}

FrameProcessor::~FrameProcessor() {
  cudaSafeCall(cudaFreeHost(keypoint_sizes_));
}

std::ostream& operator << (std::ostream& s, ushort4 v) {
  s << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
  return s;
}

cv::cuda::Stream s;

void FrameProcessor::process(
    const cv::Mat src[], 
    int threshold,
    FrameData& frame_data,
    FrameDebugData* frame_debug_data) {

  nvtxRangePushA("frame_processor");

  auto t0 = std::chrono::high_resolution_clock::now();

  CudaDeviceVector<short3>* keypoints_gpu[2] = 
      { &keypoints_gpu_l_, &keypoints_gpu_r_ };
  FastGpu* fast[2] = { &fast_l_, &fast_r_ };

  cv::cuda::Stream* s = streams_;
  cv::cuda::Event*  e = events_;

  for (int i=0; i < 2; ++i) {
  /* auto t0 = std::chrono::high_resolution_clock::now(); */
   src_img_[i].upload(src[i], s[i]);

  /* auto t1 = std::chrono::high_resolution_clock::now(); */
    cv::cuda::remap(
        src_img_[i], 
        undistorted_image_gpu_[i], 
        undistort_map_x_[i],
        undistort_map_y_[i], 
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(),
        s[i]);
      
  /* auto t2 = std::chrono::high_resolution_clock::now(); */
    fast[i]->computeScores(undistorted_image_gpu_[i], threshold, s[i]);
  /* auto t3 = std::chrono::high_resolution_clock::now(); */

  /* std::cout << "Upload: " */
  /*   << " t1 = " << duration_cast<milliseconds>(t1 - t0).count() */
  /*   << " t2 = " << duration_cast<milliseconds>(t2 - t0).count() */
  /*   << " t3 = " << duration_cast<milliseconds>(t3 - t0).count() */
  /*   << std::endl; */
  }

  for (int i=0; i < 2; ++i) {
    fast[i]->downloadKpCount(s[i]);
    e[i].record(s[i]);
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  for (int i=0; i < 2; ++i) {
    e[i].waitForCompletion();
    fast[i]->extract(threshold, *keypoints_gpu[i], s[i]);
    keypoints_gpu[i]->download(keypoints_cpu_[i], keypoint_sizes_[i], s[i]);
    e[i].record(s[i]);
  }
 

  auto t2 = std::chrono::high_resolution_clock::now();
  for (int i=0; i < 2; ++i) {
    cv::cuda::integral(undistorted_image_gpu_[i], integral_image_gpu_[i], s[i]);
  }
  auto t3 = std::chrono::high_resolution_clock::now();

  for (int i=0; i < 2; ++i) {
    e[i].waitForCompletion();
    
    nvtxRangePushA("sort");
    keypoints_cpu_[i].resize(keypoint_sizes_[i]);

    sort(
        keypoints_cpu_[i].begin(), keypoints_cpu_[i].end(), 
        [](const short3& a, const short3& b) -> bool {
          return a.y < b.y || (a.y == b.y && a.x < b.x);
        });

    keypoints_gpu[i]->upload(keypoints_cpu_[i], s[i]);
    nvtxRangePop();

    freak_.describe(
        integral_image_gpu_[i], 
        *keypoints_gpu[i],
        keypoints_cpu_[i].size(),
        descriptors_gpu_[i],
        s[i]);

    e[i].record(s[i]);
  }

  nvtxRangePushA("compute_pairs");
  computeKpPairs_(keypoints_cpu_[0], keypoints_cpu_[1], keypoint_pairs_);
  nvtxRangePop();
  keypoint_pairs_gpu_.upload(keypoint_pairs_, s[2]);

  s[2].waitEvent(e[0]);
  s[2].waitEvent(e[1]);

  matcher_.computeScores(
      descriptors_gpu_[0],
      descriptors_gpu_[1],
      keypoint_pairs_gpu_,
      keypoint_pairs_.size(),
      s[2]);

  int n_left = keypoints_cpu_[0].size();
  int n_right = keypoints_cpu_[1].size();

  s[2].waitForCompletion();

  matcher_.gatherMatches(
      n_left, n_right,
      keypoint_pairs_,
      0.8,
      matches_);

  auto t6 = std::chrono::high_resolution_clock::now();

  frame_data.points.resize(0);
  const auto& c = calib_->intrinsics;
  int k = 0;
  std::cout << "XCXC " << c.dr << ", " << c.cx << ", " << c.cy << std::endl;
  for (int t = 0; t < matches_.size(); ++t) {
    int i = matches_[t].x;
    int j = matches_[t].y; 
    auto& kp_l = keypoints_cpu_[0][i];
    auto& kp_r = keypoints_cpu_[1][j];

    if (kp_l.x > kp_r.x) {
      matches_[k++] = matches_[t];
      StereoPoint p;
      p.world.z = c.dr / (kp_r.x - kp_l.x);
      p.world.x = (kp_l.x - c.cx) * p.world.z / c.f;
      p.world.y = ((kp_l.y + kp_r.y)/2.0 - c.cy) * p.world.z / c.f;
      p.left = cv::Point2f(kp_l.x, kp_l.y);
      p.left_i = i;
      p.right = cv::Point2f(kp_r.x, kp_r.y);
      p.right_i = j;
      p.score = kp_l.z + kp_r.z;
      frame_data.points.push_back(p);
    }
  }

  matches_.resize(k);

  matches_gpu_.upload(matches_, s[2]);

  descriptor_tools::gatherDescriptors(
      descriptors_gpu_[0],
      descriptors_gpu_[1],
      matches_gpu_,
      matches_.size(),
      frame_data.d_left,
      frame_data.d_right,
      s[2]);

  s[2].waitForCompletion();

  auto t7 = std::chrono::high_resolution_clock::now();
  
  std::cout
    << " features = (" << n_left << ", " << n_right << ")"
    << " n_pairs = " << keypoint_pairs_.size()
    << " n_points = " << frame_data.points.size()
    << " upload = " << duration_cast<milliseconds>(t1 - t0).count()
    << " integral = " << duration_cast<milliseconds>(t3 - t2).count()
    << " total = " << duration_cast<milliseconds>(t7 - t0).count()
    << std::endl;

  nvtxRangePop();

  if (frame_debug_data != nullptr) {
    undistorted_image_gpu_[0].download(frame_debug_data->undistorted_image[0]);
    undistorted_image_gpu_[1].download(frame_debug_data->undistorted_image[1]);
  }
}

void FrameProcessor::computeKpPairs_(
    const PinnedVector<short3>& kps1,
    const PinnedVector<short3>& kps2,
    PinnedVector<ushort2>& keypoint_pairs) {
  keypoint_pairs.resize(0);
  int j0 = 0, j1 = 0;
  for (int i = 0; i < kps1.size(); ++i) {
    auto& pt1 = kps1[i];

    while (j0 < kps2.size() && kps2[j0].y < pt1.y - 2)
      ++j0;

    while (j1 < kps2.size() && kps2[j1].y <= pt1.y + 2)
      ++j1;

    for (int j = j0; j < j1; j++) {
      auto& pt2 = kps2[j];
      assert(fabs(pt1.y - pt2.y) <= 2);

      double dx = pt1.x - pt2.x;

      if (dx > -100 && dx < 100) {
        keypoint_pairs.push_back(make_ushort2(i, j));
      }
    }
  }
}

