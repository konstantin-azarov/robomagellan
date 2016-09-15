#include <opencv2/opencv.hpp>

#include "debug_renderer.hpp"
#include "frame_processor.hpp"
#include "cross_frame_processor.hpp"
#include "math3d.hpp"

DebugRenderer::DebugRenderer() {
}

void DebugRenderer::start(
    const FrameProcessor* p1,
    const FrameProcessor* p2,
    const CrossFrameProcessor* cfp) {
  cfp_ = cfp;
  p1_ = p1;
  p2_ = p2;

  const auto& left = p2_->undistortedImage(0);
  const auto& right = p2_->undistortedImage(1);

  int w = left.cols;
  int h = right.rows;

  img_.create(h, 2*w, CV_8UC3);

  cv::cvtColor(
      left, 
      img_(cv::Range::all(), cv::Range(0, w)), 
      CV_GRAY2RGB);

  cv::cvtColor(
      right, 
      img_(cv::Range::all(), cv::Range(w, 2*w)), 
      CV_GRAY2RGB);

}

void DebugRenderer::renderFeatures() {
  int w = img_.cols/2;

  for (int t = 0; t < 2; ++t) {
    const auto& keypoints = p2_->keypoints(t);
    int n = keypoints.size();

    for (int i = 0; i < n; ++i) {
      const auto& pt = keypoints[i].pt;

      cv::circle(img_, cv::Point(pt.x + w*t, pt.y), 3, cv::Scalar(0, 255, 0));
    }
  }
}

void DebugRenderer::renderMatches() {
  int w = img_.cols/2;
  const auto& kp_l = p2_->keypoints(0);
  const auto& kp_r = p2_->keypoints(1);
  const auto& matches_l = p2_->matches(0);
  const auto& matches_r = p2_->matches(1);

  for (int i=0; i < kp_l.size(); ++i) {
    int j = matches_l[i];
    if (i == matches_r[j]) {
      auto pl = kp_l[i].pt;
      auto pr = cv::Point(kp_r[j].pt.x + w, kp_r[j].pt.y);
      cv::circle(img_, pl, 3, cv::Scalar(0, 255, 0));
      cv::circle(img_, pr, 3, cv::Scalar(0, 255, 0));

      cv::line(img_, pl, pr, cv::Scalar(0, 255, 0));
    }
  }
}

void DebugRenderer::renderPointFeatures(int p) {
  int w = img_.cols/2;
  auto f = p2_->features(p);

  cv::circle(img_, f.first, 3, cv::Scalar(255, 0, 0));
  cv::circle(
      img_, cv::Point(f.second.x + w, f.second.y), 3, cv::Scalar(255, 0, 0));
}

void DebugRenderer::renderCrossMatches() {
  int w = img_.cols/2;

  for (const auto& f : cfp_->reprojectionFeatures()) {
    cv::circle(img_, f.s2l, 3, cv::Scalar(0, 255, 0));
    auto p2 = f.s2r + cv::Point2d(w, 0);
    cv::circle(img_, p2, 3, cv::Scalar(0, 255, 0));
    cv::line(img_, f.s2l, p2, cv::Scalar(0, 255, 0));
  }
}

void DebugRenderer::renderText(const std::string& text) {
   cv::putText(
      img_, 
      text.c_str(), 
      cv::Point(0, img_.rows - 10), 
      cv::FONT_HERSHEY_PLAIN, 
      1.0, 
      cv::Scalar(0, 255, 0));

}

std::ostream& operator << (std::ostream& s, const cv::KeyPoint& kp) {
  s << "[" << kp.pt.x << ", " << kp.pt.y << "], " 
    << "a=" << kp.angle << ", " 
    << "o=" << kp.octave << ", "
    << "r=" << kp.response << ", "
    << "s=" << kp.size;

  return s;
}

void DebugRenderer::selectKeypoint(int x, int y) {
  std::cout << "Debug [x=" << x << ", y=" << y << "]" << std::endl;  

  int w = img_.cols/2;

  int t = x >= w ? 1 : 0;
  const auto& kps = p2_->keypoints(t);

  int best = -1;
  double best_d = 1E+100;
  for (int i=0; i < kps.size(); ++i) {
    double d = norm2(cv::Point2f(x - t*w, y) - kps[i].pt);
    if (d < best_d) {
      best_d = d;
      best = i;
    }
  }

  if (best == -1) {
    std::cout << "Not found!" << std::endl;
    return;
  }

  const auto& kp = kps[best];

  std::cout << best << ": " << kp << std::endl;

  auto p1 = cv::Point(kp.pt.x + w*t, kp.pt.y);
  cv::circle(img_, p1, 3, cv::Scalar(0, 0, 255));

  int match = p2_->matches(t)[best]; 

  if (match != -1) {
    const auto& kp2 = p2_->keypoints(1-t)[match];
    std::cout << "match = " << match << ": " << kp2 << std::endl;

    auto p2 = cv::Point(kp2.pt.x + w*(1-t), kp2.pt.y);
    cv::circle(img_, p2, 3, cv::Scalar(0, 0, 255));
    cv::line(img_, p1, p2, cv::Scalar(0, 0, 255));

    if (p2_->matches(1-t)[match] == best) {
      const auto& pkp = p2_->pointKeypoints(); 
      int left_f = t == 0 ? best : match;
      int point_i = -1;
      for (int i=0; i < pkp.size(); ++i) {
        if (pkp[i] == left_f) {
          point_i = i;
          break;
        }
      }

      std::cout << "Point: " << p2_->points()[point_i] << std::endl;
    }
  }
}


