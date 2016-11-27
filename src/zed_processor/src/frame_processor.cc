#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "calibration_data.hpp"
#include "frame_processor.hpp"
#include "math3d.hpp"

using namespace std::chrono;


FrameProcessor::FrameProcessor(const StereoCalibrationData& calib) : 
    calib_(&calib),
    freak_(64.980350),
    fast_(kMaxKeypoints*5, freak_.borderWidth()),
    matcher_(kMaxKeypoints, kMaxKeypoints * 15),
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
}

std::ostream& operator << (std::ostream& s, ushort4 v) {
  s << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
  return s;
}

void FrameProcessor::process(
    const cv::Mat src[], 
    int threshold,
    FrameData& frame_data) {
  auto t0 = std::chrono::high_resolution_clock::now();

  CudaDeviceVector<short3>*
    keypoints_gpu[2] = { &keypoints_gpu_l_, &keypoints_gpu_r_ };

  for (int i=0; i < 2; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();

    src_img_[i].upload(src[i]);

    cv::cuda::remap(
        src_img_[i], 
        undistorted_image_gpu_[i], 
        undistort_map_x_[i],
        undistort_map_y_[i], 
        cv::INTER_LINEAR);

    auto t2 = std::chrono::high_resolution_clock::now();
   
    fast_.detect(undistorted_image_gpu_[i], threshold, *keypoints_gpu[i]);
    keypoints_gpu[i]->download(keypoints_cpu_[i]);

    sort(
        keypoints_cpu_[i].begin(), keypoints_cpu_[i].end(), 
        [this, i](const short3& a, const short3& b) -> bool {
          return a.y < b.y || (a.y == b.y && a.x < b.x);
        });

    keypoints_gpu[i]->upload(keypoints_cpu_[i]);

    auto t3 = std::chrono::high_resolution_clock::now();

    freak_.describe(
        undistorted_image_gpu_[i], 
        *keypoints_gpu[i],
        keypoints_cpu_[i].size(),
        descriptors_gpu_[i],
        cv::cuda::Stream::Null());

    cv::cuda::Stream::Null().waitForCompletion();

    auto t4 = std::chrono::high_resolution_clock::now();

    std::cout 
      << " kp = " << keypoints_cpu_[i].size()
      << " remap = " << duration_cast<milliseconds>(t2 - t1).count()
      << " detect = " << duration_cast<milliseconds>(t3 - t2).count() 
      << " extract = " << duration_cast<milliseconds>(t4 - t3).count();
  }

  auto t5 = std::chrono::high_resolution_clock::now();

  computeKpPairs_(keypoints_cpu_[0], keypoints_cpu_[1], keypoint_pairs_);

  int n_left = keypoints_cpu_[0].size();
  int n_right = keypoints_cpu_[1].size();

  keypoint_pairs_gpu_.upload(keypoint_pairs_);

  matcher_.computeScores(
      descriptors_gpu_[0],
      descriptors_gpu_[1],
      keypoint_pairs_gpu_,
      keypoint_pairs_.size(),
      cv::cuda::Stream::Null());

  matcher_.gatherMatches(
      n_left, n_right,
      keypoint_pairs_,
      0.8,
      matches_);

  auto t6 = std::chrono::high_resolution_clock::now();

  descriptors_gpu_[0].rowRange(0, n_left).download(
      frame_data.descriptors_left.rowRange(0, n_left));
  descriptors_gpu_[1].rowRange(0, n_right).download(
      frame_data.descriptors_right.rowRange(0, n_right));

  frame_data.points.resize(0);
  const auto& c = calib_->intrinsics;
  for (int t = 0; t < matches_.size(); ++t) {
    int i = matches_[t][0];
    int j = matches_[t][1]; 
    auto& kp_l = keypoints_cpu_[0][i];
    auto& kp_r = keypoints_cpu_[1][j];

    if (kp_l.x > kp_r.x) {
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

  auto t7 = std::chrono::high_resolution_clock::now();
  
  std::cout
    << " n_pairs = " << keypoint_pairs_.size()
    << " n_points = " << frame_data.points.size()
    << " match = " << duration_cast<milliseconds>(t6 - t5).count() 
    << " transform = " << duration_cast<milliseconds>(t7 - t6).count() 
    << " total = " << duration_cast<milliseconds>(t7 - t0).count()
    << std::endl;
}

void FrameProcessor::computeKpPairs_(
    const std::vector<short3>& kps1,
    const std::vector<short3>& kps2,
    std::vector<ushort2>& keypoint_pairs) {
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

void FrameProcessor::computeMatches_(
        const cv::Mat_<uchar>& d1,
        const cv::Mat_<uchar>& d2,
        const std::vector<short2>& keypoint_pairs,
        std::vector<ushort4> matches[2]) {

  matches[0].resize(d1.rows);
  std::fill(matches[0].begin(), matches[0].end(), make_ushort4(0xffff, 0xffff, 0xffff, 0));
  matches[1].resize(d2.rows);
  std::fill(matches[1].begin(), matches[1].end(), make_ushort4(0xffff, 0xffff, 0xffff, 0));

  for (int i=0; i < keypoint_pairs.size(); ++i) {
    const auto& p = keypoint_pairs[i];
    int score = descriptorDist(d1.row(p.x), d2.row(p.y));
    {
      ushort4 m = matches[0][p.x];
      if (score < m.x) {
        matches[0][p.x] = make_ushort4(score, m.x, p.y, 0);
      } else if (score < m.y) {
        matches[0][p.x] = make_ushort4(m.x, score, m.z, 0);
      }
    }
    {
      ushort4 m = matches[1][p.y];
      if (score < m.x) {
        matches[1][p.y] = make_ushort4(score, m.x, p.x, 0);
      } else if (score < m.y) {
        matches[1][p.y] = make_ushort4(m.x, score, m.z, 0);
      }
    }
  }

  for (int t = 0; t < 2; ++t) {
    for (auto& m : matches[t]) {
      if (m.x != 0xFFFF) {
        if (m.x / static_cast<double>(m.y) >= 0.8) {
          m.z = 0xFFFF;
        }
      }
    }
  }
}


