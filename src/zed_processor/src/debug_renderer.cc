#include <algorithm>

#include <opencv2/opencv.hpp>

#include "debug_renderer.hpp"
#include "frame_processor.hpp"
#include "cross_frame_processor.hpp"
#include "math3d.hpp"

DebugRenderer::DebugRenderer(
    const FrameProcessor* p1,
    const FrameProcessor* p2,
    const CrossFrameProcessor* cfp,
    const DirectionTracker* direction_tracker,
    const MonoCalibrationData* mono_calibration,
    int max_width, int max_height) 
    : p1_(p1),
      p2_(p2),
      cfp_(cfp),
      dt_(direction_tracker),
      mono_calibration_(mono_calibration),
      max_width_(max_width), 
      max_height_(max_height) {
  assert(max_width_ % 2 == 0);
  assert(max_height_ % 2 == 0);

  const auto& left1 = p1_->undistortedImage(0);

  w_ = left1.cols;
  h_ = left1.rows;

  scale_ = 1.0;
  if (2*w_ > max_width_) {
    scale_ = max_width_/(2.0*w_);
  }

  if (2*h_ > max_height_) {
    scale_ = std::min(scale_, max_height_/(2.0*h_));
  }
}

bool DebugRenderer::loop() {
  bool done = false,
       next_frame = false;

  bool stereo_mode = true;

  bool show_features = false, 
       show_matches = false,
       show_cross_matches = false,
       show_filtered_matches = false,
       show_clique = false,
       show_clique_cross_features = false,
       show_inlier_features = false;

  while (!done) {
    if (stereo_mode) {
      renderStereo();

      if (show_features) {
        renderFeatures();
      }

      if (show_matches) {
        renderMatches();
      }

      if (show_cross_matches) {
        renderAllCrossMatches();
      }

      if (show_filtered_matches) {
        renderFilteredCrossMatches();
      }

      if (show_clique) {
        renderCliqueMatches();
      }

      if (show_clique_cross_features) {
        renderCliqueFeatures();
      }

      if (show_inlier_features) {
        renderInlierFeatures();
      }
    } else {
      renderTarget();

      if (show_matches) {
        renderTargetMatches(dt_->reprojectionFeatures());
      }

      if (show_inlier_features) {
        renderTargetMatches(dt_->bestSample());
      }
    }

    cv::Mat scaled_image;
    cv::resize(img_, scaled_image, cv::Size(), scale_, scale_);
    cv::imshow("debug", scaled_image);
    
    int key = cv::waitKey(0);
    if (key != -1) {
      key &= 0xFF;
    }
    switch(key) {
      case 't':
        stereo_mode = !stereo_mode;
        break;
      case 'f':
        show_features = !show_features;
        break;
      case 'm':
        show_matches = !show_matches;
        break;
      case 'x':
        show_cross_matches = !show_cross_matches;
        break;
      case 'z':
        show_filtered_matches = !show_filtered_matches;
        break;
      case 'c':
        show_clique = !show_clique;
        break;
      case '1':
        show_clique_cross_features = !show_clique_cross_features;
        break;
      case '2':
        show_inlier_features = !show_inlier_features;
        break;
      case 'n':
        next_frame = true;
        done = true;
        break;
      case 27:
        done = true;
        break;
    }
  }

  return next_frame;
}

void DebugRenderer::renderStereo() {
  img_.create(2*h_, 2*w_, CV_8UC3);

  drawImage_(
      p1_->undistortedImage(0), img_(cv::Range(0, h_), cv::Range(0, w_)));
  drawImage_(
      p1_->undistortedImage(1), img_(cv::Range(0, h_), cv::Range(w_, 2*w_)));
  drawImage_(
      p2_->undistortedImage(0), img_(cv::Range(h_, 2*h_), cv::Range(0, w_)));
  drawImage_(
      p2_->undistortedImage(1), img_(cv::Range(h_, 2*h_), cv::Range(w_, 2*w_)));
}

void DebugRenderer::renderTarget() {
  img_.create(h_, 2*w_, CV_8UC3);

  drawImage_(
      p1_->undistortedImage(0), img_(cv::Range(0, h_), cv::Range(0, w_)));

  cv::Mat target_img_raw = cv::imread(
      dt_->target()->image_file,
      cv::IMREAD_GRAYSCALE);
  cv::Mat target_img;

  cv::remap(
      target_img_raw,
      target_img,
      mono_calibration_->undistort_maps.x,
      mono_calibration_->undistort_maps.y,
      cv::INTER_LINEAR);
  drawImage_(
      target_img, img_(cv::Range(0, h_), cv::Range(w_, 2*w_))); 
}

void DebugRenderer::renderFeatures() {
  int row = 0;

  for (auto* p : { p1_, p2_ }) {
    for (int t = 0; t < 2; ++t) {
      const auto& keypoints = p->keypoints(t);
      int n = keypoints.size();

      for (int i = 0; i < n; ++i) {
        const auto& pt = keypoints[i];

        cv::Point2f p(pt.x, pt.y);

        cv::circle(
            img_, 
            p + cv::Point2f(w_*t, h_*row), 
            3, 
            cv::Scalar(0, 255, 0));
      }
    }

    row++;
  }
}

void DebugRenderer::renderMatches() {
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
        
      auto pl = f.first + cv::Point2f(0, row*h_);
      auto pr = f.second + cv::Point2f(w_, row*h_);
      cv::circle(img_, pl, 3, color);
      cv::circle(img_, pr, 3, color);
    }

    row++;
  }
}

void DebugRenderer::renderTargetMatches(
    const std::vector<MonoReprojectionFeature>& features) {
  cv::Scalar color(0, 255, 0);

  for (const auto& f : features) {
    cv::line(
        img_,
        f.s1,
        f.s2 + cv::Point2d(w_),
        color,
        1);
    cv::circle(img_, f.s1, 3, cv::Scalar(0, 0, 255));
    cv::circle(img_, f.s2 + cv::Point2d(w_), 3, cv::Scalar(0, 0, 255));
  }
}

void DebugRenderer::renderPointFeatures(int p) {
  auto f = p2_->features(p);

  cv::circle(img_, f.first, 3, cv::Scalar(255, 0, 0));
  cv::circle(
      img_, cv::Point(f.second.x + w_, f.second.y), 3, cv::Scalar(255, 0, 0));
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
  cv::Scalar green(0, 255, 0), red(0, 0, 255);

  auto R = cfp_->rot();
  auto tm = cfp_->t();

  std::cout << "Clique errors:";

  for (const auto& f : features) {
    auto p1 = projectPoint(cfp_->calibration().intrinsics, R.t()*(f.r2 - tm));
    auto p2 = projectPoint(cfp_->calibration().intrinsics, R*f.r1 + tm);

    drawCross_(f.s1l, green);
    drawCross_(f.s2l + cv::Point2d(0, h_), green);
    cv::circle(img_, p1.first, 5, red);
    cv::circle(img_, p1.second + cv::Point2d(w_, 0), 5, red);
    cv::line(img_, f.s1l, f.s2l + cv::Point2d(0, h_), green);
    cv::line(img_, f.s1l, p1.first, red);
    cv::line(
        img_, 
        f.s1r + cv::Point2d(w_, 0), 
        p1.second + cv::Point2d(w_, 0), 
        red);

    drawCross_(f.s1r + cv::Point2d(w_, 0), green);
    drawCross_(f.s2r + cv::Point2d(w_, h_), green);
    cv::line(
        img_, 
        f.s1r + cv::Point2d(w_, 0), 
        f.s2r + cv::Point2d(w_, h_), green);
    cv::circle(img_, p2.first + cv::Point2d(0, h_), 5, red);
    cv::circle(img_, p2.second + cv::Point2d(w_, h_), 5, red);

    std::cout << " " << f.error;
  }

  std::cout << std::endl;
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
  int side = u >= w_ ? 1 : 0;
  int frame = v >= h_ ? 1 : 0;

  auto p = frame == 0 ? p1_ : p2_;

  const auto& kps = p->keypoints(side);

  int x = std::round((u - side*w_)/scale_);
  int y = std::round((v - frame*h_)/scale_);

  int best = -1;
  double best_d = 1E+100;
  for (int i=0; i < kps.size(); ++i) {
    double d = norm2(cv::Point2f(x, y) - cv::Point2f(kps[i].x, kps[i].y));
    if (d < best_d) {
      best_d = d;
      best = i;
    }
  }

  if (best == -1) {
    std::cout << "Not found!" << std::endl;
    return;
  }

  cv::Point2f kp(kps[best].x, kps[best].y);

  std::cout << best << ": " << kp << std::endl;

  auto p1 = kp*scale_ + cv::Point2f(side*w_, frame*h_);
  cv::circle(img_, p1, 3, cv::Scalar(0, 0, 255));

  int match = p->matches(side)[best]; 

  if (match != -1) {
    const auto& s2 = p->keypoints(1-side)[match];
    cv::Point2f kp2(s2.x, s2.y);
    std::cout << "match = " << match << ": " << kp2 << std::endl;

    auto p2 = kp2*scale_ + cv::Point2f((1-side)*w_, frame*h_);
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
    auto p1l = p1_->features(m.i1).first + cv::Point2f(0, 0);
    auto p1r = p1_->features(m.i1).second + cv::Point2f(w_, 0);
    auto p2l = p2_->features(m.i2).first + cv::Point2f(0, h_);
    auto p2r = p2_->features(m.i2).second + cv::Point2f(w_, h_);

    cv::circle(img_, p1l, 3, cv::Scalar(0, 255, 0));
    cv::circle(img_, p1r, 3, cv::Scalar(0, 255, 0));
    cv::circle(img_, p2l, 3, cv::Scalar(0, 255, 0));
    cv::circle(img_, p2r, 3, cv::Scalar(0, 255, 0));
    cv::line(img_, p1l, p2l, cv::Scalar(0, 255, 0));
    cv::line(img_, p1r, p2r, cv::Scalar(0, 255, 0));
}
