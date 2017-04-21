#include <chrono>

#include "cross_frame_processor.hpp"
#include "frame_processor.hpp"
#include "math3d.hpp"

#include "descriptor_tools.hpp"

using namespace std::chrono;

namespace e = Eigen; 

std::ostream& operator << (std::ostream& s, const e::Vector3d& v) {
  s << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
  return s;
}

std::ostream& operator << (std::ostream& s, const e::Vector2d& v) {
  s << "(" << v.x() << ", " << v.y() << ")";
  return s;
}

CrossFrameProcessor::CrossFrameProcessor(
    const StereoCalibrationData& calibration,
    const CrossFrameProcessorConfig& config) 
  : calibration_(calibration), 
    reprojection_estimator_(&calibration.intrinsics),
    config_(config),
    new_descriptors_gpu_(config.maxTrackedFeatures()),
    matcher_(config_.maxTrackedFeatures(), config_.max_incoming_features) {

    int n = config_.maxTrackedFeatures();
    int m = config_.max_incoming_features;

    tracked_features_.reserve(n);
    new_features_.reserve(n);
    tmp_features_.reserve(n);

    tracked_descriptors_.reset(new Stereo<cv::cudev::GpuMat_<uint8_t>>(
        n, FreakGpu::kDescriptorWidth)); 
    back_desc_buffer_.reset(new Stereo<cv::cudev::GpuMat_<uint8_t>>(
        n, FreakGpu::kDescriptorWidth)); 

    features_with_matches_.reserve(n);
    clique_features_.reserve(n);
    near_features_.reserve(n);
    far_features_.reserve(n);
    sorted_features_.reserve(n);

    new_descriptors_.l.reserve(n);
    new_descriptors_.r.reserve(n);

    scores_gpu_.l.create(n, m);
    scores_gpu_.r.create(n, m); 
    scores_cpu_.l.create(n, m);
    scores_cpu_.r.create(n, m);
}

bool CrossFrameProcessor::process(
    const FrameData& p, 
    Eigen::Quaterniond& r, 
    Eigen::Vector3d& t,
    Eigen::Matrix3d* t_cov,
    CrossFrameDebugData* debug_data) {
  auto t0 = std::chrono::high_resolution_clock::now();

  fillNewFeatures_(p);

  int n_f = tracked_features_.size();
  int n_p = new_features_.size();

  if (n_p == 0) {
    return false;
  }

  std::cout << "Ego: " << n_f << " " << n_p << std::endl;

  bool valid_estimate = true;

  auto t1 = std::chrono::high_resolution_clock::now();

  if (n_f > 0) {
    descriptor_tools::scores(
        tracked_descriptors_->l.rowRange(0, n_f),
        p.d_left.rowRange(0, n_p),
        scores_gpu_.l,
        cv::cuda::Stream::Null());

    descriptor_tools::scores(
        tracked_descriptors_->r.rowRange(0, n_f),
        p.d_right.rowRange(0, n_p), 
        scores_gpu_.r,
        cv::cuda::Stream::Null());

    cv::Mat_<uint16_t> scores_l = 
      scores_cpu_.l(cv::Range(0, n_f), cv::Range(0, n_p));
    cv::Mat_<uint16_t> scores_r =
      scores_cpu_.r(cv::Range(0, n_f), cv::Range(0, n_p));

    scores_gpu_.l(cv::Range(0, n_f), cv::Range(0, n_p)).download(scores_l);
    scores_gpu_.r(cv::Range(0, n_f), cv::Range(0, n_p)).download(scores_r);

    matchStereo_(scores_l, scores_r);

    features_with_matches_.resize(0);
    near_features_.resize(0);
    far_features_.resize(0);

    for (auto& f : tracked_features_) {
      if (f.match != nullptr) {
        features_with_matches_.push_back(&f);
        if (isNear_(f)) {
          near_features_.push_back(&f);
        } else if (isFar_(f)) {
          far_features_.push_back(&f);
        }
      }
    }

    std::cout << "Near = " << near_features_.size() 
              << ", Far = " << far_features_.size() << std::endl;

    selectInitialEstimationFeatures_();
    std::cout << "Estimation = " << estimation_features_.size() << std::endl;


    if (!estimatePose_(r, t, t_cov)) {
      valid_estimate = false;
    }

    if (valid_estimate) {
      selectFinalEstimationFeatures_();
      std::cout << "Final = " << estimation_features_.size() << std::endl;
      valid_estimate = estimatePose_(r, t, t_cov);
    }
  } else {
    valid_estimate = false;
  }

  if (debug_data != nullptr) {
    debug_data->old_tracked_features = tracked_features_;
    debug_data->new_tracked_features = new_features_;

    for (auto& f : debug_data->old_tracked_features) {
      if (f.match != nullptr) {
        f.match = &debug_data->new_tracked_features[
          f.match - &debug_data->new_tracked_features.front()];
      }
    }

    for (auto& f : debug_data->new_features) {
      f.match = nullptr;
    }

    debug_data->near_features.clear();
    for (auto* f : near_features_) {
      debug_data->near_features.insert(
          &debug_data->old_tracked_features[f - &tracked_features_.front()]);
    }

    debug_data->far_features.clear();
    for (auto* f : far_features_) {
      debug_data->far_features.insert(
          &debug_data->old_tracked_features[f - &tracked_features_.front()]);
    }
  }

  updateFeatures_(p, r, t);

  if (debug_data != nullptr) {
    debug_data->new_features = tracked_features_;
  }

  auto t4 = std::chrono::high_resolution_clock::now();

  std::cout << "Odometry stats: " << std::endl;
  std::cout << "  tracked = " << tracked_features_.size() << std::endl;

 /* std::cout */ 
  /*   << "matches = " << full_matches_.size() */ 
/* //    s< " clique = " << clique_reprojection_features_.size() */ 
  /*   << "; t:" */
  /*   << " setup = " << duration_cast<milliseconds>(t1 - t0).count() */
  /*   << " match = " << duration_cast<milliseconds>(t2 - t1).count() */
  /*   << " clique = " << duration_cast<milliseconds>(t3 - t2).count() */
  /*   << " estimate = " << duration_cast<milliseconds>(t4 - t3).count() */
  /*   << std::endl; */


  return valid_estimate;
}

void CrossFrameProcessor::fillNewFeatures_(const FrameData& d) {
  new_features_.clear();

  for (const auto& p : d.points) {
    WorldFeature f;
    
    f.w = e::Vector3d(p.world.x, p.world.y, p.world.z);
    f.left = e::Vector2d(p.left.x, p.left.y);
    f.right = e::Vector2d(p.right.x, p.right.y);
    f.score = p.score;
    f.age = 0;
    f.bucket_id = 0;
    f.match = nullptr;

    /* std::cout << "N: " << f.w << " " << f.left << " " << f.right << std::endl; */

    new_features_.push_back(f);
  }
}

void CrossFrameProcessor::matchStereo_(
    const cv::Mat_<uint16_t>& scores_l, const cv::Mat_<uint16_t>& scores_r) {
  matchMono_(scores_l, matches_.l);
  matchMono_(scores_r, matches_.r);

  int n = tracked_features_.size();

  for (int i = 0; i < n; ++i) {
    if (matches_.l[i] == matches_.r[i]) {
      tracked_features_[i].match = &new_features_[matches_.l[i]];

      /* std::cout << "M: " << i << " -> " << matches_.l[i] << std::endl; */

      new_features_[matches_.l[i]].match = &tracked_features_[i];
    }
  }
}

void CrossFrameProcessor::matchMono_(
    const cv::Mat_<uint16_t>& scores,
    std::vector<int>& matches) {

  int n = scores.rows;
  int m = scores.cols;

  matches.resize(n);
  std::fill(std::begin(matches), std::end(matches), -1);

  for (int i = 0; i < n; ++i) {
    int best_v = 1024;
    int best_j = -1;
    for (int j = 0; j < m; ++j) {
      if (scores(i, j) < best_v) {
        best_v = scores(i, j);
        best_j = j;
      }
    }

    matches[i] = best_j;
    /* std::cout << "MM:" << i << " -> " << matches[i] << std::endl; */
  }
}

void CrossFrameProcessor::updateFeatures_(
    const FrameData& p,
    const e::Quaterniond& r, const e::Vector3d& t) {
  auto r_inv = r.inverse();
  double bucket_w = calibration_.raw.size.width / config_.x_buckets;
  double bucket_h = calibration_.raw.size.height / config_.y_buckets;
  // translate features, remove matches, update age and remove stale features
  // update bucket index
  sorted_features_.clear();

  for (int i = 0; i < features_with_errors_.size(); ++i) {
    auto& f = features_with_errors_[i];

    if (f.err > config_.max_reprojection_error) {
      continue;
    }

    f.f->age = std::max(0, f.f->age) + 1;

    f.f->w = r_inv * (f.f->w - t);
    std::tie(f.f->left, f.f->right) = projectPoint(
        calibration_.intrinsics, f.f->w);

    sorted_features_.push_back(f.f);
  }

  for (auto& f : new_features_) {
    if (f.match == nullptr) {
      sorted_features_.push_back(&f);
    }
  }

  for (auto* f : sorted_features_) {
    int hor_idx = f->left.x() / bucket_w;
    int vert_idx = f->left.y() / bucket_h;
    f->bucket_id = vert_idx * config_.x_buckets + hor_idx;
  }

  std::sort(
      std::begin(sorted_features_),
      std::end(sorted_features_),
      [](const WorldFeature* a, const WorldFeature* b) {
        if (a->bucket_id != b->bucket_id) {
          return a->bucket_id < b->bucket_id;
        } else if (a->age != b->age) {
          return a->age > b->age;
        } else {
          return a->score > b->score;
        }
      });

  tmp_features_.clear();
  new_descriptors_.l.clear();
  new_descriptors_.r.clear();

  int current_bucket = -1, current_count = 0;
  for (const auto* f : sorted_features_) {
    if (current_bucket != f->bucket_id) {
      current_bucket = f->bucket_id;
      current_count = 0;
    }

    if (current_count < config_.max_tracked_features_per_bucket) {
      current_count++;
      tmp_features_.push_back(*f);

      if (f->age != 0) {
        int i = f - &tracked_features_.front();
        new_descriptors_.l.push_back(tracked_descriptors_->l[i]);
        new_descriptors_.r.push_back(tracked_descriptors_->r[i]);
      } else {
        int i = f - &new_features_.front();
        new_descriptors_.l.push_back(p.d_left[i]);
        new_descriptors_.r.push_back(p.d_right[i]);
      }
    }
  }

  std::swap(tmp_features_, tracked_features_);

  new_descriptors_gpu_.l.upload(new_descriptors_.l);
  new_descriptors_gpu_.r.upload(new_descriptors_.r);

  descriptor_tools::gatherDescriptors(
      new_descriptors_gpu_.l, 
      new_descriptors_.l.size(),
      back_desc_buffer_->l,
      cv::cuda::Stream::Null());
  descriptor_tools::gatherDescriptors(
      new_descriptors_gpu_.r, 
      new_descriptors_.r.size(),
      back_desc_buffer_->r,
      cv::cuda::Stream::Null());
      
  std::swap(back_desc_buffer_, tracked_descriptors_);
}

void CrossFrameProcessor::selectInitialEstimationFeatures_() {
  estimation_features_.clear();
  buildCliqueNear_();
  buildCliqueFar_();
}

void CrossFrameProcessor::selectFinalEstimationFeatures_() {
  sorted_features_.clear();
  for (int i = 0; i < features_with_errors_.size(); ++i) {
    auto& f = features_with_errors_[i];

    if (f.err <= config_.max_reprojection_error) {
      sorted_features_.push_back(f.f);
    }
  }

  std::sort(
      std::begin(sorted_features_),
      std::end(sorted_features_),
      [](const WorldFeature* a, const WorldFeature* b) {
        if (a->bucket_id != b->bucket_id) {
          return a->bucket_id < b->bucket_id;
        } else if (a->age != b->age) {
          return a->age > b->age;
        } else {
          return a->score > b->score;
        }
      });

  estimation_features_.clear();
  int current_bucket = -1, current_count = 0;
  for (auto* f : sorted_features_) {
    if (current_bucket != f->bucket_id) {
      current_bucket = f->bucket_id;
      current_count = 0;
    }

    if (current_count < config_.max_estimation_features_per_bucket) {
      estimation_features_.push_back(f);
      current_count++;
    }
  }
}

bool CrossFrameProcessor::isNear_(const WorldFeature& f) const {
  return f.w.z() < config_.near_feature_threshold;

}

bool CrossFrameProcessor::isFar_(const WorldFeature& f) const {
  float dz = 60 / 3.6 * 1000 / 30;   // 60 km/h -> mm/sec -> per frame
  double dx = 0.1 * dz;
  double fc = calibration_.intrinsics.f;
  double cx = calibration_.intrinsics.cx;
  double cy = calibration_.intrinsics.cy;
  
  double z = f.w.z();

  double u1 = fc * f.w.x() / (z - dz) + cx;
  double v1 = fc * f.w.y() / (z - dz) + cy;
  double u2 = fc * (f.w.x() + dx) / z + cx;
  double v2 = fc * (f.w.y() + dx) / z + cy;

  return fabs(f.left.x() - u1) < 2 && fabs(f.left.y() - v1) < 2 &&
         fabs(f.left.x() - u2) < 2 && fabs(f.left.y() - v2) < 2;
};

void CrossFrameProcessor::buildCliqueNear_() {
  // Construct the matrix of the matches
  // matches i and j have an edge is distance between corresponding points in the
  // first image is equal to the distance between corresponding points in the
  // last image.
  int n = near_features_.size();
  
  if (n == 0) {
    return;
  }

  clique_.reset(n);

  for (int i = 0; i < n; ++i) {
    clique_.setColor(i, near_features_[i]->bucket_id);
  }

  for (int i = 0; i < n; ++i) {
    auto* m1 = near_features_[i];
    for (int j = 0; j < i; ++j) {
      auto& m2 = near_features_[j];
      double l1 = (m1->w - m1->match->w).norm();
      double l2 = (m2->w - m2->match->w).norm();
     
      double dl1 = deltaL(m1->w, m1->match->w);
      double dl2 = deltaL(m2->w, m2->match->w);

      if (std::abs(l1 - l2) < std::min(3*std::sqrt(dl1*dl1 + dl2*dl2), 2000.0)) {
        clique_.addEdge(i, j);
      }
    }
  }

  const std::vector<int>& clique = clique_.compute();

  for (int i : clique) {
    estimation_features_.push_back(near_features_[i]);
  }
}

void CrossFrameProcessor::buildCliqueFar_() {
  int n = far_features_.size();

  if (n == 0) {
    return;
  }

  clique_.reset(n);

  for (int i = 0; i < n; ++i) {
    clique_.setColor(i, far_features_[i]->bucket_id);
  }

  for (int i = 0; i < n; ++i) {
    auto* m1 = far_features_[i];
    Eigen::Vector3d cp1 = m1->w.normalized().cross(m1->match->w.normalized());

    for (int j = 0; j < i; ++j) {
      auto* m2 = far_features_[j]; 
      Eigen::Vector3d cp2 = 
        m2->w.normalized().cross(m2->match->w.normalized().normalized());

      if (fabs(cp1.norm() - cp2.norm()) < M_PI/180*5) {
        clique_.addEdge(i, j);
      }
    }
  }

  const std::vector<int>& clique = clique_.compute();
  
  for (int i : clique) {
    estimation_features_.push_back(far_features_[i]);
  }
}

inline double sqr(double v) {
  return v*v;
}

double CrossFrameProcessor::deltaL(const e::Vector3d& p1, 
                                   const e::Vector3d& p2) {
  double x1 = p1.x(), y1 = p1.y(), z1 = p1.z();
  double x2 = p2.x(), y2 = p2.y(), z2 = p2.z();

  double f = calibration_.intrinsics.f;
  double t = calibration_.intrinsics.dr/f; 

  double L = (p1 - p2).norm();

  double A = sqr((x1 - x2)*(t - x1) - (y1 - y2)*y1 - (z1 - z2)*z1);
  double B = sqr(((x1 - x2)*x1 + (y1 - y2)*y1 + (z1 - z2)*z1));
  double C = 0.5*sqr(t*(y1 - y2));
  double D = sqr((x1 - x2)*(t - x2) - (y1 - y2)*y2 - (z1 - z2)*z2);
  double E = sqr((x1 - x2)*x2 + (y1 - y2)*y2 + (z1 - z2)*z2);
  double F = 0.5*sqr(t*(y1 - y2));

  return 0.1/(L*f*t)*std::sqrt(z1*z1*(A+B+C) + z2*z2*(D+E+F));
}

void CrossFrameProcessor::fillReprojectionFeatures_() {
  reprojection_features_.clear();
  
  for (const auto* ef : estimation_features_) {
    StereoReprojectionFeature rf;

    rf.r2 = ef->w;
    rf.s2l = ef->left;
    rf.s2r = ef->right;
    
    rf.r1 = ef->match->w; 
    rf.s1l = ef->match->left;
    rf.s1r = ef->match->right;

    /* std::cout << "F: " << rf.r2 << " " << rf.s2l << " " << rf.s2r */ 
              /* << " " << rf.r1 << " " << rf.s1l << " " << rf.s1r << std::endl; */ 

    reprojection_features_.push_back(rf);
  }

  std::cout << "Reprojection features: " << reprojection_features_.size() 
    << std::endl;
}

double CrossFrameProcessor::fillReprojectionErrors_(
    const Eigen::Quaterniond& r, 
    const Eigen::Vector3d& tm) {
  double total = 0;

  features_with_errors_.clear();

  auto ri = r.inverse();

  for (auto* f : features_with_matches_) {
    auto p1 = projectPoint(calibration_.intrinsics, r*f->match->w + tm);
    auto p2 = projectPoint(calibration_.intrinsics, ri*(f->w - tm)); 

    double e1 = (p1.first - f->left).squaredNorm() + 
      (p1.second - f->right).squaredNorm(); 
    double e2 = (p2.first - f->match->left).squaredNorm() + 
      (p2.second - f->match->right).squaredNorm();
    
    features_with_errors_.push_back(FeatureWithError {f, e1 + e2});
  }

  return 0;
}



bool CrossFrameProcessor::estimatePose_(
    Eigen::Quaterniond& r, 
    Eigen::Vector3d& t,
    Eigen::Matrix3d* t_cov) {

  t.setZero();
  r.setIdentity();

  if (estimation_features_.size() < config_.min_features_for_estimation) {
    features_with_errors_.clear();
    return false;
  }

  fillReprojectionFeatures_();
  reprojection_estimator_.estimate(reprojection_features_, r, t, t_cov);
  fillReprojectionErrors_(r, t);

  // xcxc
  return true;

}
