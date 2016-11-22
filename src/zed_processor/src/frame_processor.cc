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
    fast_(50000, freak_.borderWidth()),
    freak_(64.980350) {
  for (int i=0; i < 2; ++i) {
    undistort_map_x_[i].upload(calib_->undistort_maps[i].x);
    undistort_map_y_[i].upload(calib_->undistort_maps[i].y);
  }
}

std::ostream& operator << (std::ostream& s, ushort4 v) {
  s << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
  return s;
}

void FrameProcessor::process(const cv::Mat src[], int threshold) {
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i=0; i < 2; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();

    src_img_[i].upload(src[i]);

    cv::cuda::remap(
        src_img_[i], 
        undistorted_image_gpu_[i], 
        undistort_map_x_[i],
        undistort_map_y_[i], 
        cv::INTER_LINEAR);

    undistorted_image_gpu_[i].download(undistorted_image_[i]);

    auto t2 = std::chrono::high_resolution_clock::now();
   
    keypoints_[i].clear();

    fast_.detect(undistorted_image_gpu_[i], threshold);
    fast_.keypoints().download(keypoints_[i]);

    sort(
        keypoints_[i].begin(), keypoints_[i].end(), 
        [this, i](const short3& a, const short3& b) -> bool {
          return a.y < b.y || (a.y == b.y && a.x < b.x);
        });

    fast_.keypoints().upload(keypoints_[i]);

    auto t3 = std::chrono::high_resolution_clock::now();

    freak_.describe(undistorted_image_gpu_[i], fast_.keypoints());

    descriptors_[i].create(freak_.descriptors().size(), CV_8UC1);

    freak_.descriptors().download(descriptors_[i]);

    auto t4 = std::chrono::high_resolution_clock::now();

    std::cout 
      << " kp = " << keypoints_[i].size()
      << " remap = " << duration_cast<milliseconds>(t2 - t1).count()
      << " detect = " << duration_cast<milliseconds>(t3 - t2).count() 
      << " extract = " << duration_cast<milliseconds>(t4 - t3).count();
  }

  auto t5 = std::chrono::high_resolution_clock::now();

  computeKpPairs_(keypoints_[0], keypoints_[1], keypoint_pairs_);

  computeMatches_(descriptors_[0], descriptors_[1], keypoint_pairs_, matches_);

  auto t6 = std::chrono::high_resolution_clock::now();

  points_.resize(0);
  point_keypoints_.resize(0);
  for (int i=0; i < matches_[0].size(); ++i) {
    int j = matches_[0][i].z;
    if (j != 0xFFFF && matches_[1][j].z == i) {
      auto& kp1 = keypoints_[0][i];
      auto& kp2 = keypoints_[1][j];

      if (kp1.x - kp2.x > 0) {
        points_.push_back(
            cv::Point3d(
              kp1.x, (kp1.y + kp2.y)/2, std::max(0.0f, (float)(kp1.x - kp2.x))));
        point_keypoints_.push_back(i);
      }
    }
  }

  if (points_.size() > 0) {
    cv::perspectiveTransform(points_, points_, calib_->Q);
  }

  auto t7 = std::chrono::high_resolution_clock::now();
  
  std::cout
    << " n_pairs = " << keypoint_pairs_.size()
    << " n_points = " << points_.size()
    << " match = " << duration_cast<milliseconds>(t6 - t5).count() 
    << " transform = " << duration_cast<milliseconds>(t7 - t6).count() 
    << " total = " << duration_cast<milliseconds>(t7 - t0).count()
    << std::endl;
}

void FrameProcessor::computeKpPairs_(
    const std::vector<short3>& kps1,
    const std::vector<short3>& kps2,
    std::vector<short2>& keypoint_pairs) {
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

      double dx = pt2.x - pt1.x;

      if (dx > -100 && dx < 100) {
        keypoint_pairs.push_back(make_short2(i, j));
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

void FrameProcessor::match(
           const std::vector<short3>& kps1,
           const cv::Mat& desc1,
           const std::vector<short3>& kps2,
           const cv::Mat& desc2,
           int inv,
           std::vector<int>& matches) {
  matches.resize(kps1.size());

  int j0 = 0, j1 = 0;
  int pairs = 0;

  for (int i = 0; i < kps1.size(); ++i) {
    auto& pt1 = kps1[i];

    matches[i] = -1;

    while (j0 < kps2.size() && kps2[j0].y < pt1.y - 2)
      ++j0;

    while (j1 < kps2.size() && kps2[j1].y <= pt1.y + 2)
      ++j1;

    assert(j1 >= j0);

    double best_d = 1E+15, second_d = 1E+15;
    double best_j = -1;

    for (int j = j0; j < j1; j++) {
      auto& pt2 = kps2[j];

      assert(fabs(pt1.y - pt2.y) <= 2);

      double dx = inv*(pt1.x - pt2.x);

      if (dx > -100 && dx < 100) {
        double dist = descriptorDist(desc1.row(i), desc2.row(j));
        if (dist < best_d) {
          second_d = best_d;
          best_d = dist;
          best_j = j;
        } else if (dist < second_d) {
          second_d = dist;
        }
        pairs++;
      }
    }
   
    //std::cout << best_d << " " << second_d << std::endl;

    if (best_j > -1  && best_d / second_d < 0.8) {
      matches[i] = best_j;
    }
  }
}


