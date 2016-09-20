#include <algorithm>

#include <opencv2/opencv.hpp>

#include "debug_renderer.hpp"
#include "frame_processor.hpp"
#include "cross_frame_processor.hpp"
#include "math3d.hpp"

DebugRenderer::DebugRenderer(int max_width, int max_height) :
    max_width_(max_width), max_height_(max_height) {
  assert(max_width_ % 2 == 0);
  assert(max_height_ % 2 == 0);
}

void DebugRenderer::start(
    const FrameProcessor* p1,
    const FrameProcessor* p2,
    const CrossFrameProcessor* cfp) {
  cfp_ = cfp;
  p1_ = p1;
  p2_ = p2;

  const auto& left1 = p1_->undistortedImage(0);

  int w = left1.cols;
  int h = left1.rows;

  scale_ = 1.0;
  if (2*w > max_width_) {
    scale_ = max_width_/(2.0*w);
  }

  if (2*h > max_height_) {
    scale_ = std::min(scale_, max_height_/(2.0*h));
  }

  w = (int)(scale_*w);
  h = (int)(scale_*h);
  
  img_.create(2*h, 2*w, CV_8UC3);

  drawImage_(
      left1, img_(cv::Range(0, h), cv::Range(0, w)));
  drawImage_(
      p1_->undistortedImage(1), img_(cv::Range(0, h), cv::Range(w, 2*w)));
  drawImage_(
      p2_->undistortedImage(0), img_(cv::Range(h, 2*h), cv::Range(0, w)));
  drawImage_(
      p2_->undistortedImage(1), img_(cv::Range(h, 2*h), cv::Range(w, 2*w)));
}

void DebugRenderer::renderFeatures() {
  int w = img_.cols/2;
  int h = img_.rows/2;
  int row = 0;

  for (auto* p : { p1_, p2_ }) {
    for (int t = 0; t < 2; ++t) {
      const auto& keypoints = p->keypoints(t);
      int n = keypoints.size();

      for (int i = 0; i < n; ++i) {
        const auto& pt = keypoints[i].pt;

        cv::circle(
            img_, 
            pt * scale_ + cv::Point2f(w*t, h*row), 
            3, 
            cv::Scalar(0, 255, 0));
      }
    }

    row++;
  }
}

void DebugRenderer::renderMatches() {
  int w = img_.cols/2;
  int h = img_.rows/2;
  int row = 0;

  for (const auto* p : { p1_, p2_ }) {
    const auto& pts = p->points();

    for (int i=0; i < pts.size(); ++i) {
      auto pt = pts[i];
      auto f = p->features(i);

      double d = cv::norm(pt);

      cv::Scalar color;
      if (d < 500) {
        color = cv::Scalar(0, 255, 255);
      } else if (d > 20 * 1000) {
        color = cv::Scalar(0, 0, 255);
      } else {
        color = cv::Scalar(0, 255, 0);
      }
        
      auto pl = f.first * scale_ + cv::Point2f(0, row*h);
      auto pr = f.second * scale_ + cv::Point2f(w, row*h);
      cv::circle(img_, pl, 3, color);
      cv::circle(img_, pr, 3, color);
    }

    row++;
  }
}

void DebugRenderer::renderPointFeatures(int p) {
  int w = img_.cols/2;
  auto f = p2_->features(p);

  cv::circle(img_, f.first, 3, cv::Scalar(255, 0, 0));
  cv::circle(
      img_, cv::Point(f.second.x + w, f.second.y), 3, cv::Scalar(255, 0, 0));
}

void DebugRenderer::renderAllCrossMatches() {
  for (auto m : cfp_->fullMatches()) {
    drawMatch_(m);
  }
}

void DebugRenderer::renderFilteredCrossMatches() {
  const auto& full_matches = cfp_->fullMatches();
  for (auto i : cfp_->filteredMatches()) {
    drawMatch_(full_matches[i]);
  }
}

void DebugRenderer::renderCliqueMatches() {
  const auto& full_matches = cfp_->fullMatches();
  for (auto i : cfp_->clique()) {
    drawMatch_(full_matches[i]);
  }
}

void DebugRenderer::drawCross_(const cv::Point& pt, const cv::Scalar& color) {
  cv::line(img_, cv::Point(pt.x - 5, pt.y), cv::Point(pt.x + 5, pt.y), color);
  cv::line(img_, cv::Point(pt.x, pt.y - 5), cv::Point(pt.x, pt.y + 5), color);
}

void DebugRenderer::dumpCrossMatches(const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  fs << "matches" << "[";
  for (const auto& m : cfp_->fullMatches()) {
    fs << "{" << "p1" << m.p1 << "p2" << m.p2 << "}";
  }
  fs << "]";

  std::cout << "Dumped cross-frame matches to " << filename << std::endl;
}

void DebugRenderer::renderCliqueFeatures() {
  renderReprojectionFeatures(cfp_->cliqueFeatures());
}

void DebugRenderer::renderInlierFeatures() {
  renderReprojectionFeatures(cfp_->inlierFeatures());
}

void DebugRenderer::renderReprojectionFeatures(
    const std::vector<ReprojectionFeatureWithError>& features) {
  int w = img_.cols/2, h = img_.rows/2;

  cv::Scalar green(0, 255, 0), red(0, 0, 255);

  auto R = cfp_->rot();
  auto tm = cfp_->t();

  std::cout << "Clique errors:" << std::endl;

  for (const auto& f : features) {
    auto p1 = projectPoint(cfp_->calibration().intrinsics, R.t()*(f.r2 - tm));
    auto p2 = projectPoint(cfp_->calibration().intrinsics, R*f.r1 + tm);

    drawCross_(f.s1l * scale_, green);
    drawCross_(f.s2l * scale_ + cv::Point2d(0, h), green);
    cv::circle(img_, p1.first * scale_, 5, red);
    cv::circle(img_, p1.second * scale_ + cv::Point2d(w, 0), 5, red);
    cv::line(img_, f.s1l * scale_, f.s2l * scale_ + cv::Point2d(0, h), green);
    cv::line(img_, f.s1l * scale_, p1.first * scale_, red);
    cv::line(
        img_, 
        f.s1r * scale_ + cv::Point2d(w, 0), 
        p1.second * scale_ + cv::Point2d(w, 0), 
        red);

    drawCross_(f.s1r * scale_ + cv::Point2d(w, 0), green);
    drawCross_(f.s2r * scale_ + cv::Point2d(w, h), green);
    cv::line(
        img_, 
        f.s1r * scale_ + cv::Point2d(w, 0), 
        f.s2r * scale_ + cv::Point2d(w, h), green);
    cv::circle(img_, p2.first * scale_ + cv::Point2d(0, h), 5, red);
    cv::circle(img_, p2.second * scale_ + cv::Point2d(w, h), 5, red);

    std::cout << " " << f.error << std::endl;
  }
}

void DebugRenderer::dumpClique(const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  fs << "clique" << "[";
  for (int m : cfp_->clique()) {
    fs << m;
  }
  fs << "]";
  
  std::cout << "Dumped cross-frame clique to " << filename << std::endl;
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

void DebugRenderer::selectKeypoint(int u, int v) {
  int w = img_.cols/2;
  int h = img_.rows/2;

  int side = u >= w ? 1 : 0;
  int frame = v >= h ? 1 : 0;

  auto p = frame == 0 ? p1_ : p2_;

  const auto& kps = p->keypoints(side);

  int x = std::round((u - side*w)/scale_);
  int y = std::round((v - frame*h)/scale_);

  int best = -1;
  double best_d = 1E+100;
  for (int i=0; i < kps.size(); ++i) {
    double d = norm2(cv::Point2f(x, y) - kps[i].pt);
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

  auto p1 = kp.pt*scale_ + cv::Point2f(side*w, frame*h);
  cv::circle(img_, p1, 3, cv::Scalar(0, 0, 255));

  int match = p->matches(side)[best]; 

  if (match != -1) {
    const auto& kp2 = p->keypoints(1-side)[match];
    std::cout << "match = " << match << ": " << kp2 << std::endl;

    auto p2 = kp2.pt*scale_ + cv::Point2f((1-side)*w, frame*h);
    cv::circle(img_, p2, 3, cv::Scalar(0, 0, 255));
    cv::line(img_, p1, p2, cv::Scalar(0, 0, 255));

    if (p->matches(1-side)[match] == best) {
      const auto& pkp = p->pointKeypoints(); 
      int left_f = side == 0 ? best : match;
      int point_i = -1;
      for (int i=0; i < pkp.size(); ++i) {
        if (pkp[i] == left_f) {
          point_i = i;
          break;
        }
      }

      std::cout << "Point: " << p->points()[point_i] << std::endl;
    }
  }
}

void DebugRenderer::drawImage_(const cv::Mat& src, const cv::Mat& dest) {
  cv::Mat tmp;

  cv::cvtColor(src, tmp, CV_GRAY2RGB);
  cv::resize(tmp, dest, dest.size());
}

void DebugRenderer::drawMatch_(const CrossFrameMatch& m) {
    int w = img_.cols/2;
    int h = img_.rows/2;

    auto p1l = p1_->features(m.i1).first * scale_ + cv::Point2f(0, 0);
    auto p1r = p1_->features(m.i1).second * scale_ + cv::Point2f(w, 0);
    auto p2l = p2_->features(m.i2).first * scale_ + cv::Point2f(0, h);
    auto p2r = p2_->features(m.i2).second * scale_ + cv::Point2f(w, h);

    cv::circle(img_, p1l, 3, cv::Scalar(0, 255, 0));
    cv::circle(img_, p1r, 3, cv::Scalar(0, 255, 0));
    cv::circle(img_, p2l, 3, cv::Scalar(0, 255, 0));
    cv::circle(img_, p2r, 3, cv::Scalar(0, 255, 0));
    cv::line(img_, p1l, p2l, cv::Scalar(0, 255, 0));
    cv::line(img_, p1r, p2r, cv::Scalar(0, 255, 0));
}
