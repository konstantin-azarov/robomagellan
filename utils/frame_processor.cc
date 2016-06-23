#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "calibration_data.hpp"
#include "frame_processor.hpp"
#include "math3d.hpp"
#include "utils.hpp" 

struct Match {
  int leftIndex, rightIndex;
};

FrameProcessor::FrameProcessor(const CalibrationData& calib) : calib_(&calib) {
}

void FrameProcessor::process(const cv::Mat src[], int threshold) {
  for (int i=0; i < 2; ++i) {
    cv::remap(
        src[i], 
        undistorted_image_[i], 
        calib_->undistortMaps[i].x, 
        calib_->undistortMaps[i].y, 
        cv::INTER_LINEAR);

//    auto detector = cv::BRISK::create(threshold);
    auto extractor = cv::xfeatures2d::FREAK::create();
    auto detector = cv::xfeatures2d::SURF::create();
//    auto extractor = detector;
    double t0 = nanoTime();
    detector->detect(undistorted_image_[i], keypoints_[i]);
    extractor->compute(undistorted_image_[i], keypoints_[i], descriptors_[i]);
    double t1 = nanoTime();
    std::cout << "dT = " << t1 - t0 << std::endl;

    order_[i].resize(keypoints_[i].size());
    for (int j=0; j < order_[i].size(); ++j) {
      order_[i][j] = j;
    }
    
    sort(order_[i].begin(), order_[i].end(), [this, i](int a, int b) -> bool {
        auto& k1 = keypoints_[i][a].pt;
        auto& k2 = keypoints_[i][b].pt;

        return (k1.y < k2.y || (k1.y == k2.y && k1.x < k2.x));
    });
  }

  for (int i=0; i < 2; ++i) {
    int j = 1 - i;
    match(
        keypoints_[i], order_[i], descriptors_[i],
        keypoints_[j], order_[j], descriptors_[j],
        i ? -1 : 1,
        matches_[i]);
  }

  points_.resize(0);
  point_keypoints_.resize(0);
  match_points_.resize(matches_[0].size());
  for (int i=0; i < matches_[0].size(); ++i) {
    int j = matches_[0][i];
    match_points_[i] = -1;
    if (j != -1 && matches_[1][j] == i) {
      auto& kp1 = keypoints_[0][i].pt;
      auto& kp2 = keypoints_[1][j].pt;
      points_.push_back(
          cv::Point3d(kp1.x, (kp1.y + kp2.y)/2, std::max(0.0f, kp1.x - kp2.x)));
      point_keypoints_.push_back(i);
      match_points_[i] = points_.size() - 1;
    }
  }

  if (points_.size() > 0) {
    cv::perspectiveTransform(points_, points_, calib_->Q);
  }
}

void FrameProcessor::match(const std::vector<cv::KeyPoint>& kps1,
           const std::vector<int>& idx1,
           const cv::Mat& desc1,
           const std::vector<cv::KeyPoint>& kps2,
           const std::vector<int>& idx2,
           const cv::Mat& desc2,
           int inv,
           std::vector<int>& matches) {
  matches.resize(kps1.size());

  int j0 = 0, j1 = 0;

  for (int i : idx1) {
    auto& pt1 = kps1[i].pt;

    matches[i] = -1;

    while (j0 < kps2.size() && kps2[idx2[j0]].pt.y < pt1.y - 2)
      ++j0;

    while (j1 < kps2.size() && kps2[idx2[j1]].pt.y < pt1.y + 2)
      ++j1;

    assert(j1 >= j0);

    double best_d = 1E+15, second_d = 1E+15;
    double best_j = -1;

    for (int jj = j0; jj < j1; jj++) {
      int j = idx2[jj];
      auto& pt2 = kps2[j].pt;

      assert(fabs(pt1.y - pt2.y) <= 2);

      double dx = inv*(pt1.x - pt2.x);

      if (dx > 3 && dx < 100) {
        double dist = descriptorDist(desc1.row(i), desc2.row(j));
        if (dist < best_d) {
          best_d = dist;
          best_j = j;
        } else if (dist < second_d) {
          second_d = dist;
        }
      }
    }
   
    //std::cout << best_d << " " << second_d << std::endl;

    if (best_j > -1  && best_d / second_d < 0.8) {
      matches[i] = best_j;
    }
  }
}


