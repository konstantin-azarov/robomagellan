#include <algorithm>

#include <opencv2/opencv.hpp>

#include "debug_renderer.hpp"
#include "frame_processor.hpp"
#include "cross_frame_processor.hpp"
#include "math3d.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

DebugRenderer::DebugRenderer(
    const StereoCalibrationData& calib,
    const FrameData& f1,
    const FrameDebugData& d1,
    const FrameData& f2,
    const FrameDebugData& d2,
    const CrossFrameDebugData& cfd,
    Eigen::Affine3d* ground_truth,
    int max_width, int max_height) 
    : calib_(calib),
      f1_(f1), fd1_(d1),
      f2_(f2), fd2_(d2),
      cfd_(cfd),
      ground_truth_(ground_truth),
      max_width_(max_width), 
      max_height_(max_height) {
  assert(max_width_ % 2 == 0);
  assert(max_height_ % 2 == 0);

  const auto& left1 = fd1_.undistorted_image[0];

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
       show_all_reprojection_features = false,
       show_clique = false,
       show_clique_cross_features = false,
       show_inlier_features = false,
       use_ground_truth = false;


  while (!done) {
    if (stereo_mode) {
      renderStereo();

/*       if (show_features) { */
/*         renderFeatures(); */
/*       } */

      if (show_matches) {
        renderMatches();
      }

      if (show_all_reprojection_features) {
        renderReprojectionFeatures(
            cfd_.all_reprojection_features,
            use_ground_truth ? *ground_truth_ : cfd_.t_inlier_);
      }

      if (show_clique_cross_features) {
        renderReprojectionFeatures(
            cfd_.clique_reprojection_features,
            use_ground_truth ? *ground_truth_ : cfd_.t_clique_);
      }

      if (show_inlier_features) {
        renderReprojectionFeatures(
            cfd_.inlier_reprojection_features,
            use_ground_truth ? *ground_truth_ : cfd_.t_inlier_);
      }
    }
/*     } else { */
/*       renderTarget(); */

/*       if (show_matches) { */
/*         renderTargetMatches(dt_->reprojectionFeatures()); */
/*       } */

/*       if (show_inlier_features) { */
/*         renderTargetMatches(dt_->bestSample()); */
/*       } */
/*     } */

    cv::Mat scaled_image;
    cv::resize(img_, scaled_image, cv::Size(), scale_, scale_);
    cv::imshow("debug", scaled_image);
    
    int key = cv::waitKey(0);
    if (key != -1) {
      key &= 0xFF;
    }
    switch(key) {
/*       case 't': */
/*         stereo_mode = !stereo_mode; */
/*         break; */
/*       case 'f': */
/*         show_features = !show_features; */
/*         break; */
      case 'm':
        show_matches = !show_matches;
        break;
/*       case 'x': */
/*         show_cross_matches = !show_cross_matches; */
/*         break; */
/*       case 'z': */
/*         show_filtered_matches = !show_filtered_matches; */
/*         break; */
/*       case 'c': */
/*         show_clique = !show_clique; */
/*         break; */
      case '0':
        show_all_reprojection_features = !show_all_reprojection_features;
        break;
      case '1':
        show_clique_cross_features = !show_clique_cross_features;
        break;
      case '2':
        show_inlier_features = !show_inlier_features;
        break;
      case 'g':
        use_ground_truth = !use_ground_truth && ground_truth_ != nullptr;
        break;
/*       case 'n': */
/*         next_frame = true; */
/*         done = true; */
/*         break; */
/*       case 's': */
/*         cv::imwrite("/tmp/left.png", p1_->undistortedImage(0)); */
/*         break; */
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
      fd1_.undistorted_image[0], img_(cv::Range(0, h_), cv::Range(0, w_)));
  drawImage_(
      fd1_.undistorted_image[1], img_(cv::Range(0, h_), cv::Range(w_, 2*w_)));
  drawImage_(
      fd2_.undistorted_image[0], img_(cv::Range(h_, 2*h_), cv::Range(0, w_)));
  drawImage_(
      fd2_.undistorted_image[1], img_(cv::Range(h_, 2*h_), cv::Range(w_, 2*w_)));
}

/* void DebugRenderer::renderTarget() { */
/*   img_.create(h_, 2*w_, CV_8UC3); */

/*   drawImage_( */
/*       p1_->undistortedImage(0), img_(cv::Range(0, h_), cv::Range(0, w_))); */

/*   cv::Mat target_img_raw = cv::imread( */
/*       dt_->target()->image_file, */
/*       cv::IMREAD_GRAYSCALE); */
/*   cv::Mat target_img; */

/*   cv::remap( */
/*       target_img_raw, */
/*       target_img, */
/*       mono_calibration_->undistort_maps.x, */
/*       mono_calibration_->undistort_maps.y, */
/*       cv::INTER_LINEAR); */
/*   drawImage_( */
/*       target_img, img_(cv::Range(0, h_), cv::Range(w_, 2*w_))); */ 
/* } */

/* void DebugRenderer::renderFeatures() { */
/*   int row = 0; */

/*   for (auto* p : { p1_, p2_ }) { */
/*     for (int t = 0; t < 2; ++t) { */
/*       const auto& keypoints = p->keypoints(t); */
/*       int n = keypoints.size(); */

/*       for (int i = 0; i < n; ++i) { */
/*         const auto& pt = keypoints[i]; */

/*         cv::Point2f p(pt.x, pt.y); */

/*         cv::circle( */
/*             img_, */ 
/*             p + cv::Point2f(w_*t, h_*row), */ 
/*             3, */ 
/*             cv::Scalar(0, 255, 0)); */
/*       } */
/*     } */

/*     row++; */
/*   } */
/* } */

void DebugRenderer::renderMatches() {
  int row = 0;

  for (const auto& frame : { f1_, f2_ }) {
    const auto& pts = frame.points;

    for (int i=0; i < pts.size(); ++i) {
      auto pt = pts[i];

      double d = cv::norm(pt.world);

      cv::Scalar color;
      if (d < 500) {
        color = cv::Scalar(0, 255, 255);
      } else if (d > 20 * 1000) {
        color = cv::Scalar(0, 0, 255);
      } else {
        color = cv::Scalar(0, 255, 0);
      }
        
      auto pl = pt.left + cv::Point2f(0, row*h_);
      auto pr = pt.right + cv::Point2f(w_, row*h_);
      cv::circle(img_, pl, 3, color);
      cv::circle(img_, pr, 3, color);
    }

    row++;
  }
}

/* void DebugRenderer::renderTargetMatches( */
/*     const std::vector<MonoReprojectionFeature>& features) { */
/*   cv::Scalar color(0, 255, 0); */

/*   for (const auto& f : features) { */
/*     cv::line( */
/*         img_, */
/*         f.s1, */
/*         f.s2 + cv::Point2d(w_), */
/*         color, */
/*         1); */
/*     cv::circle(img_, f.s1, 3, cv::Scalar(0, 0, 255)); */
/*     cv::circle(img_, f.s2 + cv::Point2d(w_), 3, cv::Scalar(0, 0, 255)); */
/*   } */
/* } */

/* void DebugRenderer::renderPointFeatures(int p) { */
/*   auto f = p2_->features(p); */

/*   cv::circle(img_, f.first, 3, cv::Scalar(255, 0, 0)); */
/*   cv::circle( */
/*       img_, cv::Point(f.second.x + w_, f.second.y), 3, cv::Scalar(255, 0, 0)); */
/* } */

/* void DebugRenderer::renderAllCrossMatches() { */
/*   for (auto m : cfp_->fullMatches()) { */
/*     drawMatch_(m); */
/*   } */
/* } */

/* void DebugRenderer::renderFilteredCrossMatches() { */
/*   const auto& full_matches = cfp_->fullMatches(); */
/*   for (auto i : cfp_->filteredMatches()) { */
/*     drawMatch_(full_matches[i]); */
/*   } */
/* } */

/* void DebugRenderer::renderCliqueMatches() { */
/*   const auto& full_matches = cfp_->fullMatches(); */
/*   for (auto i : cfp_->clique()) { */
/*     drawMatch_(full_matches[i]); */
/*   } */
/* } */

/* void DebugRenderer::drawCross_(const cv::Point& pt, const cv::Scalar& color) { */
/*   cv::line(img_, cv::Point(pt.x - 5, pt.y), cv::Point(pt.x + 5, pt.y), color); */
/*   cv::line(img_, cv::Point(pt.x, pt.y - 5), cv::Point(pt.x, pt.y + 5), color); */
/* } */

/* void DebugRenderer::dumpCrossMatches(const std::string& filename) { */
/*   cv::FileStorage fs(filename, cv::FileStorage::WRITE); */

/*   fs << "matches" << "["; */
/*   for (const auto& m : cfp_->fullMatches()) { */
/*     fs << "{" << "p1" << m.p1 << "p2" << m.p2 << "}"; */
/*   } */
/*   fs << "]"; */

/*   std::cout << "Dumped cross-frame matches to " << filename << std::endl; */
/* } */

void DebugRenderer::renderReprojectionFeatures(
    const std::vector<ReprojectionFeatureWithError>& features,
    Eigen::Affine3d t) {

  // r2 is from the first frame
  // r1 is from the second frame
  // r2 = t*r1

  cv::Scalar green(0, 255, 0), red(0, 0, 255), yellow(0, 255, 255);

  auto draw_feature = [&](Eigen::Vector2d f, Eigen::Vector2d p) {
    cv::line(
        img_,
        cv::Point2d(f.x(), f.y()),
        cv::Point2d(p.x(), p.y()),
        yellow);
    cv::line(
        img_, 
        cv::Point2d(f.x() - 5, f.y()), 
        cv::Point2d(f.x() + 5, f.y()), 
        green);
    cv::line(
        img_, 
        cv::Point2d(f.x(), f.y() - 5), 
        cv::Point2d(f.x(), f.y() + 5), 
        green);
    cv::circle(
        img_,
        cv::Point2d(p.x(), p.y()),
        5,
        red);
  };

  auto right = Eigen::Vector2d(w_, 0);
  auto bottom = Eigen::Vector2d(0, h_);

  for (const auto& f : features) {
    auto p1 = projectPoint(calib_.intrinsics, t.inverse() * f.r2);
    auto p2 = projectPoint(calib_.intrinsics, t*f.r1);

    draw_feature(f.s2l, p2.first);
    draw_feature(f.s2r + right, p2.second + right);

    draw_feature(f.s1l + bottom, p1.first + bottom);
    draw_feature(f.s1r + bottom + right, p1.second + bottom + right);
  }
}

/* void DebugRenderer::dumpClique(const std::string& filename) { */
/*   cv::FileStorage fs(filename, cv::FileStorage::WRITE); */

/*   fs << "clique" << "["; */
/*   for (int m : cfp_->clique()) { */
/*     fs << m; */
/*   } */
/*   fs << "]"; */
  
/*   std::cout << "Dumped cross-frame clique to " << filename << std::endl; */
/* } */

/* void DebugRenderer::renderText(const std::string& text) { */
/*    cv::putText( */
/*       img_, */ 
/*       text.c_str(), */ 
/*       cv::Point(0, img_.rows - 10), */ 
/*       cv::FONT_HERSHEY_PLAIN, */ 
/*       1.0, */ 
/*       cv::Scalar(0, 255, 0)); */
/* } */

/* std::ostream& operator << (std::ostream& s, const cv::KeyPoint& kp) { */
/*   s << "[" << kp.pt.x << ", " << kp.pt.y << "], " */ 
/*     << "a=" << kp.angle << ", " */ 
/*     << "o=" << kp.octave << ", " */
/*     << "r=" << kp.response << ", " */
/*     << "s=" << kp.size; */

/*   return s; */
/* } */

/* void DebugRenderer::selectKeypoint(int u, int v) { */
/*   int side = u >= w_ ? 1 : 0; */
/*   int frame = v >= h_ ? 1 : 0; */

/*   auto p = frame == 0 ? p1_ : p2_; */

/*   const auto& kps = p->keypoints(side); */

/*   int x = std::round((u - side*w_)/scale_); */
/*   int y = std::round((v - frame*h_)/scale_); */

/*   int best = -1; */
/*   double best_d = 1E+100; */
/*   for (int i=0; i < kps.size(); ++i) { */
/*     double d = norm2(cv::Point2f(x, y) - cv::Point2f(kps[i].x, kps[i].y)); */
/*     if (d < best_d) { */
/*       best_d = d; */
/*       best = i; */
/*     } */
/*   } */

/*   if (best == -1) { */
/*     std::cout << "Not found!" << std::endl; */
/*     return; */
/*   } */

/*   cv::Point2f kp(kps[best].x, kps[best].y); */

/*   std::cout << best << ": " << kp << std::endl; */

/*   auto p1 = kp*scale_ + cv::Point2f(side*w_, frame*h_); */
/*   cv::circle(img_, p1, 3, cv::Scalar(0, 0, 255)); */

/*   int match = p->matches(side)[best]; */ 

/*   if (match != -1) { */
/*     const auto& s2 = p->keypoints(1-side)[match]; */
/*     cv::Point2f kp2(s2.x, s2.y); */
/*     std::cout << "match = " << match << ": " << kp2 << std::endl; */

/*     auto p2 = kp2*scale_ + cv::Point2f((1-side)*w_, frame*h_); */
/*     cv::circle(img_, p2, 3, cv::Scalar(0, 0, 255)); */
/*     cv::line(img_, p1, p2, cv::Scalar(0, 0, 255)); */

/*     if (p->matches(1-side)[match] == best) { */
/*       const auto& pkp = p->pointKeypoints(); */ 
/*       int left_f = side == 0 ? best : match; */
/*       int point_i = -1; */
/*       for (int i=0; i < pkp.size(); ++i) { */
/*         if (pkp[i] == left_f) { */
/*           point_i = i; */
/*           break; */
/*         } */
/*       } */

/*       std::cout << "Point: " << p->points()[point_i] << std::endl; */
/*     } */
/*   } */
/* } */

void DebugRenderer::drawImage_(const cv::Mat& src, const cv::Mat& dest) {
  cv::Mat tmp;

  cv::cvtColor(src, tmp, CV_GRAY2RGB);
  cv::resize(tmp, dest, dest.size());
}

/* void DebugRenderer::drawMatch_(const CrossFrameMatch& m) { */
/*     auto p1l = p1_->features(m.i1).first + cv::Point2f(0, 0); */
/*     auto p1r = p1_->features(m.i1).second + cv::Point2f(w_, 0); */
/*     auto p2l = p2_->features(m.i2).first + cv::Point2f(0, h_); */
/*     auto p2r = p2_->features(m.i2).second + cv::Point2f(w_, h_); */

/*     cv::circle(img_, p1l, 3, cv::Scalar(0, 255, 0)); */
/*     cv::circle(img_, p1r, 3, cv::Scalar(0, 255, 0)); */
/*     cv::circle(img_, p2l, 3, cv::Scalar(0, 255, 0)); */
/*     cv::circle(img_, p2r, 3, cv::Scalar(0, 255, 0)); */
/*     cv::line(img_, p1l, p2l, cv::Scalar(0, 255, 0)); */
/*     cv::line(img_, p1r, p2r, cv::Scalar(0, 255, 0)); */
/* } */
