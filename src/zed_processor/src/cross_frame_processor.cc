#include <chrono>

#include "cross_frame_processor.hpp"
#include "frame_processor.hpp"
#include "math3d.hpp"

#include "descriptor_tools.hpp"

using namespace std::chrono;

const int maxPoints = 4000;

CrossFrameProcessor::CrossFrameProcessor(
    const StereoCalibrationData& calibration,
    const CrossFrameProcessorConfig& config) 
  : calibration_(calibration), 
    reprojection_estimator_(&calibration.intrinsics),
    config_(config) {
  scores_left_.create(4000, 4000);
  scores_left_gpu_.create(4000, 4000);
  scores_right_.create(4000, 4000);
  scores_right_gpu_.create(4000, 4000);
}

bool CrossFrameProcessor::process(
    const FrameData& p1, 
    const FrameData& p2,
    bool use_initial_estimate,
    Eigen::Quaterniond& r, 
    Eigen::Vector3d& t,
    Eigen::Matrix3d* t_cov,
    CrossFrameDebugData* debug_data) {
  auto t0 = std::chrono::high_resolution_clock::now();

  const auto& points1 = p1.points;
  const auto& points2 = p2.points;

  int n1 = points1.size();
  int n2 = points2.size();

  auto t1 = std::chrono::high_resolution_clock::now();

  descriptor_tools::scores(
      p1.d_left.rowRange(0, n1), 
      p2.d_left.rowRange(0, n2),
      scores_left_gpu_,
      cv::cuda::Stream::Null());

  descriptor_tools::scores(
      p1.d_right.rowRange(0, n1), 
      p2.d_right.rowRange(0, n2),
      scores_right_gpu_,
      cv::cuda::Stream::Null());

  scores_left_gpu_.rowRange(0, n1).colRange(0, n2).download(
      scores_left_.rowRange(0, n1).colRange(0, n2));
  scores_right_gpu_.rowRange(0, n1).colRange(0, n2).download(
      scores_right_.rowRange(0, n1).colRange(0, n2));

  match(p1, p2, scores_left_, n1, n2, matches_left_[0]);
  match(p1, p2, scores_right_, n1, n2, matches_right_[0]);
  match(p2, p1, scores_left_.t(), n2, n1, matches_left_[1]);
  match(p2, p1, scores_right_.t(), n2, n1, matches_right_[1]);

  full_matches_.resize(0);

  auto sz = calibration_.raw.size;
  float bucket_w = static_cast<float>(sz.width) / config_.x_buckets;
  float bucket_h = static_cast<float>(sz.height) / config_.y_buckets;

  for (int i=0; i < n1; ++i) {
    int j = matches_left_[0][i];

    if (j != -1 && matches_right_[0][i] == j && 
        matches_left_[1][j] == i && matches_right_[1][j] == i) {
      const auto& kp = points2[j].left;
      int bucket = std::floor(kp.y / bucket_h) * config_.x_buckets + 
        std::floor(kp.x / bucket_w);
      float score = points1[i].score + points2[j].score;
      full_matches_.push_back(
          CrossFrameMatch(
            points1[i].world, points2[j].world, bucket, score, i, j));
    }
  }

  // Bucketing
  
  std::sort(
      full_matches_.begin(), full_matches_.end(), 
      [](const CrossFrameMatch& a, const CrossFrameMatch& b) {
        if (a.bucket_index != b.bucket_index) {
          return a.bucket_index < b.bucket_index;
        }
        return a.score > b.score;
      });

  int current_bucket = -1;
  int current_cnt = 0;
  int i = 0;
  filtered_matches_.resize(0);
  for (const auto& m : full_matches_) {
    if (m.bucket_index != current_bucket) {
      current_cnt = 0;
      current_bucket = m.bucket_index;
    }
    if (current_cnt < config_.max_features_per_bucket) {
      filtered_matches_.push_back(i);
      ++current_cnt;
    }
    ++i;
  }
  
  fillReprojectionFeatures_(p1, p2);

  if (debug_data != nullptr) {
    debug_data->all_reprojection_features = all_reprojection_features_;
  }

  // --

  auto t2 = std::chrono::high_resolution_clock::now();

  if (!use_initial_estimate) {
    buildCliqueNear_(p1, p2);
    buildCliqueFar_(p1, p2);
  }

  auto t3 = std::chrono::high_resolution_clock::now();

  if (!estimatePose_(use_initial_estimate, r, t, t_cov, debug_data)) {
    return false;
  }

  auto t4 = std::chrono::high_resolution_clock::now();
  
  std::cout 
    << "matches = " << full_matches_.size() 
//    s< " clique = " << clique_reprojection_features_.size() 
    << "; t:"
    << " setup = " << duration_cast<milliseconds>(t1 - t0).count()
    << " match = " << duration_cast<milliseconds>(t2 - t1).count()
    << " clique = " << duration_cast<milliseconds>(t3 - t2).count()
    << " estimate = " << duration_cast<milliseconds>(t4 - t3).count()
    << std::endl;


  return true;
}

void CrossFrameProcessor::match(
    const FrameData& p1, 
    const FrameData& p2,
    const cv::Mat_<ushort>& scores,
    int n1, int n2,
    std::vector<int>& matches) {

  matches.resize(n1);

  for (int i = 0; i < n1; ++i) {
    int best_j = -1;
    double best_dist = 1E+15;

    for (int j = 0; j < n2; ++j) {
      double screen_d = cv::norm(p1.points[i].left - p2.points[j].left);
      if (screen_d > config_.match_radius) {
        continue;
      }

      double d = scores[i][j];

      if (d < best_dist) {
        best_dist = d;
        best_j = j;
      }
    }

    matches[i] = best_j;
  }
}

void CrossFrameProcessor::buildCliqueNear_(
    const FrameData& p1, 
    const FrameData& p2) {
  // Construct the matrix of the matches
  // matches i and j have an edge is distance between corresponding points in the
  // first image is equal to the distance between corresponding points in the
  // last image.
  int n = full_matches_.size();
  int m = filtered_matches_.size();
  clique_.reset(n);

  for (int i = 0; i < m; ++i) {
    auto& m1 = full_matches_[filtered_matches_[i]];
    for (int j = 0; j < i; ++j) {
      auto& m2 = full_matches_[filtered_matches_[j]];
      double l1 = sqrt(norm3(m1.p1 - m2.p1));
      double l2 = sqrt(norm3(m1.p2 - m2.p2));
     
      double dl1 = deltaL(m1.p1, m2.p1);
      double dl2 = deltaL(m1.p2, m2.p2);

      if (m1.p1.z < 20000 && m1.p2.z < 20000 && 
          m2.p1.z < 20000 && m2.p2.z < 20000 &&
          std::abs(l1 - l2) < std::min(3*std::sqrt(dl1*dl1 + dl2*dl2), 2000.0)) {
        clique_.addEdge(filtered_matches_[i], filtered_matches_[j]);
      }
    }
  }

  const std::vector<int>& clique = clique_.compute();

  filtered_reprojection_features_.resize(0);
  for (int i : clique) {
    filtered_reprojection_features_.push_back(all_reprojection_features_[i]);
  }
}

void CrossFrameProcessor::buildCliqueFar_(
    const FrameData& p1, 
    const FrameData& p2) {
  // Construct the matrix of the matches
  // matches i and j have an edge is distance between corresponding points in the
  // first image is equal to the distance between corresponding points in the
  // last image.
  int n = full_matches_.size();
  int m = filtered_matches_.size();
  clique_.reset(n);

  float dz = 0.1*60/3.6 * 2000;
  int cx = calibration_.intrinsics.cx;
  int cy = calibration_.intrinsics.cy;

  auto is_far = [&](double z, cv::Point2d s) {
    return z > dz * std::max(fabs(s.x - cx), fabs(s.y - cy));
  };

  for (int i = 0; i < m; ++i) {
    auto& m1 = full_matches_[filtered_matches_[i]];
    Eigen::Vector3d cp1 = 
      Eigen::Vector3d(m1.p1.x, m1.p1.y, m1.p1.z).normalized().cross(
        Eigen::Vector3d(m1.p2.x, m1.p2.y, m1.p2.z).normalized());

    if (is_far(m1.p1.z, p1.points[m1.i1].left)) {
      for (int j = 0; j < i; ++j) {
        auto& m2 = full_matches_[filtered_matches_[j]];
        Eigen::Vector3d cp2 = 
          Eigen::Vector3d(m2.p1.x, m2.p1.y, m2.p1.z).normalized().cross(
            Eigen::Vector3d(m2.p2.x, m2.p2.y, m2.p2.z).normalized());

        if (fabs(cp1.norm() - cp2.norm()) < M_PI/180*5) {
          clique_.addEdge(filtered_matches_[i], filtered_matches_[j]);
        }
      }
    }
  }

  const std::vector<int>& clique = clique_.compute();
  
  for (int i : clique) {
    filtered_reprojection_features_.push_back(all_reprojection_features_[i]);
  }
}

inline double sqr(double v) {
  return v*v;
}

double CrossFrameProcessor::deltaL(const cv::Point3d& p1, 
                                   const cv::Point3d& p2) {
  double x1 = p1.x, y1 = p1.y, z1 = p1.z;
  double x2 = p2.x, y2 = p2.y, z2 = p2.z;

  double f = calibration_.intrinsics.f;
  double t = calibration_.intrinsics.dr/f; 

  double L = cv::norm(p1 - p2);

  double A = sqr((x1 - x2)*(t - x1) - (y1 - y2)*y1 - (z1 - z2)*z1);
  double B = sqr(((x1 - x2)*x1 + (y1 - y2)*y1 + (z1 - z2)*z1));
  double C = 0.5*sqr(t*(y1 - y2));
  double D = sqr((x1 - x2)*(t - x2) - (y1 - y2)*y2 - (z1 - z2)*z2);
  double E = sqr((x1 - x2)*x2 + (y1 - y2)*y2 + (z1 - z2)*z2);
  double F = 0.5*sqr(t*(y1 - y2));

  return 0.1/(L*f*t)*std::sqrt(z1*z1*(A+B+C) + z2*z2*(D+E+F));
}

void CrossFrameProcessor::fillReprojectionFeatures_(
    const FrameData& p1,
    const FrameData& p2) {

  auto cv_to_eigen = [](const cv::Point2d& p) {
    return Eigen::Vector2d(p.x, p.y);
  };

  all_reprojection_features_.resize(full_matches_.size());
  for (int i : filtered_matches_) {
    const auto& m = full_matches_[i];

    auto& f = all_reprojection_features_[i];

    f.r2 = Eigen::Vector3d(m.p1.x, m.p1.y, m.p1.z);
    f.r1 = Eigen::Vector3d(m.p2.x, m.p2.y, m.p2.z);
    f.s2l = cv_to_eigen(p1.points[m.i1].left);
    f.s2r = cv_to_eigen(p1.points[m.i1].right);
    f.s1l = cv_to_eigen(p2.points[m.i2].left);
    f.s1r = cv_to_eigen(p2.points[m.i2].right);
  }
}

double CrossFrameProcessor::fillReprojectionErrors_(
    const Eigen::Quaterniond& r, 
    const Eigen::Vector3d& tm,
    std::vector<ReprojectionFeatureWithError>& reprojection_features) {
  double total = 0;

  for (auto& f : reprojection_features) {
    auto p1 = projectPoint(calibration_.intrinsics, r.inverse()*(f.r2 - tm));
    auto p2 = projectPoint(calibration_.intrinsics, r*f.r1 + tm);

    double e1 = (p1.first - f.s1l).squaredNorm() + 
      (p1.second - f.s1r).squaredNorm(); 
    double e2 = (p2.first - f.s2l).squaredNorm() + 
      (p2.second - f.s2r).squaredNorm();

    f.error = e1 + e2;
    total += e1 + e2;
  }

  return total / reprojection_features.size();
}

bool CrossFrameProcessor::estimatePose_(
    bool use_initial_estimate,
    Eigen::Quaterniond& r, 
    Eigen::Vector3d& t,
    Eigen::Matrix3d* t_cov, 
    CrossFrameDebugData* debug_data) {

  if (debug_data != nullptr) {
    debug_data->reprojection_features.clear();
    debug_data->pose_estimations.clear();
  }

  double reprojection_error;

  t.setZero();
  r.setIdentity();

  for (int i=0; i < 10; ++ i) {
    if (i > 0 || !use_initial_estimate) {
      estimateOne_(filtered_reprojection_features_, r, t, t_cov);

      reprojection_error = fillReprojectionErrors_(
          r, t,
          filtered_reprojection_features_);
      
      if (debug_data != nullptr) {
        debug_data->reprojection_features.push_back(
            filtered_reprojection_features_);
        debug_data->pose_estimations.push_back(Eigen::Translation3d(t)*r);
      }
    }

    fillReprojectionErrors_(r, t, all_reprojection_features_);

    std::cout << "Step " << i << ":" 
      << " features = " << filtered_reprojection_features_.size()
      << " err = " << reprojection_error
      << std::endl;

    filtered_reprojection_features_.clear();
    for (const auto& f : all_reprojection_features_) {
      if (f.error < config_.inlier_threshold) {
        filtered_reprojection_features_.push_back(f);
      }
    }
  }


  bool success = reprojection_error < 5.0;
  if (!success) {
    std::cout << "Failure: " << reprojection_error << std::endl;
  }
  return success;
}

void CrossFrameProcessor::estimateOne_(
    const std::vector<ReprojectionFeatureWithError>& features,
    Eigen::Quaterniond& r, 
    Eigen::Vector3d& t,
    Eigen::Matrix3d* t_cov) {
  tmp_reprojection_features_.resize(0);

  if (features.size() < 5) {
    std::cout << "WARNING: not enough features" << std::endl;
    return;
  }

  for (auto f : features) {
    tmp_reprojection_features_.push_back(f);
  }

  reprojection_estimator_.estimate(tmp_reprojection_features_, r, t, t_cov);
}
