#include <chrono>

#include "cross_frame_processor.hpp"
#include "frame_processor.hpp"
#include "math3d.hpp"

using namespace std::chrono;

CrossFrameProcessor::CrossFrameProcessor(
    const StereoCalibrationData& calibration,
    const CrossFrameProcessorConfig& config) 
  : calibration_(calibration), 
    reprojection_estimator_(&calibration.intrinsics),
    config_(config) {
}

bool CrossFrameProcessor::process(
    const FrameData& p1, 
    const FrameData& p2) {
  auto t0 = std::chrono::high_resolution_clock::now();

  const auto& points1 = p1.points;
  const auto& points2 = p2.points;

  auto t1 = std::chrono::high_resolution_clock::now();

  match(p1, p2, matches_[0]);
  match(p2, p1, matches_[1]);

  full_matches_.resize(0);

  auto sz = calibration_.raw.size;
  float bucket_w = static_cast<float>(sz.width) / config_.x_buckets;
  float bucket_h = static_cast<float>(sz.height) / config_.y_buckets;

  for (int i=0; i < matches_[0].size(); ++i) {
    int j = matches_[0][i];
    if (j != -1 && matches_[1][j] == i) {
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

  // --

  auto t2 = std::chrono::high_resolution_clock::now();

  buildClique_(p1, p2);
  auto t3 = std::chrono::high_resolution_clock::now();

  if (!estimatePose()) {
    return false;
  }

  auto t4 = std::chrono::high_resolution_clock::now();
  
  std::cout 
    << "matches = " << full_matches_.size() 
    << " clique = " << clique_reprojection_features_.size() 
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
    std::vector<int>& matches) {

  const auto& points1 = p1.points;
  const auto& points2 = p2.points;

  matches.resize(points1.size());

  for (int i = 0; i < points1.size(); ++i) {
    int best_j = -1;
    double best_dist = 1E+15;

    for (int j = 0; j < points2.size(); ++j) {
      double screen_d = cv::norm(p1.points[i].left - p2.points[j].left);
      if (screen_d > config_.match_radius) {
        continue;
      }

      double d1 = descriptorDist(
          p1.descriptors_left.row(p1.points[i].left_i), 
          p2.descriptors_left.row(p2.points[j].left_i));
      double d2 = descriptorDist(
          p1.descriptors_right.row(p1.points[i].right_i), 
          p2.descriptors_right.row(p2.points[j].right_i));

      double d = std::max(d1, d2);

      if (d < best_dist) {
        best_dist = d;
        best_j = j;
      }
    }

    matches[i] = best_j;
  }
}

void CrossFrameProcessor::buildClique_(
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
    for (int j = 0; j < i; ++j) {
      auto& m1 = full_matches_[filtered_matches_[i]];
      auto& m2 = full_matches_[filtered_matches_[j]];
      double l1 = sqrt(norm3(m1.p1 - m2.p1));
      double l2 = sqrt(norm3(m1.p2 - m2.p2));
     
      double dl1 = deltaL(m1.p1, m2.p1);
      double dl2 = deltaL(m1.p2, m2.p2);

      if (m1.p1.z < 10000 && m1.p2.z < 10000 && 
          m2.p1.z < 10000 && m2.p2.z < 10000 &&
          std::abs(l1 - l2) < std::min(3*std::sqrt(dl1*dl1 + dl2*dl2), 1000.0)) {
        clique_.addEdge(filtered_matches_[i], filtered_matches_[j]);
      }
    }
  }

  const std::vector<int>& clique = clique_.compute();

  clique_reprojection_features_.resize(0);
  for (int i : clique) {
    clique_reprojection_features_.push_back(all_reprojection_features_[i]);
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

  all_reprojection_features_.resize(full_matches_.size());
  for (int i : filtered_matches_) {
    const auto& m = full_matches_[i];

    auto& f = all_reprojection_features_[i];

    f.r1 = m.p1;
    f.r2 = m.p2;
    f.s1l = p1.points[m.i1].left;
    f.s1r = p1.points[m.i1].right;
    f.s2l = p2.points[m.i2].left;
    f.s2r = p2.points[m.i2].right;
  }
}

double CrossFrameProcessor::fillReprojectionErrors_(
    const cv::Mat& R, 
    const cv::Point3d& tm,
    std::vector<ReprojectionFeatureWithError>& reprojection_features) {
  double total = 0;

  for (auto& f : reprojection_features) {
    auto p1 = projectPoint(calibration_.intrinsics, R.t()*(f.r2 - tm));
    auto p2 = projectPoint(calibration_.intrinsics, R*f.r1 + tm);

    double e1 = norm2(p1.first - f.s1l) + norm2(p1.second - f.s1r); 
    double e2 = norm2(p2.first - f.s2l) + norm2(p2.second - f.s2r);

    f.error = e1 + e2;
    total += e1 + e2;
  }

  return total / reprojection_features.size();
}

bool CrossFrameProcessor::estimatePose() {
  if (clique_reprojection_features_.size() < 
        config_.min_features_for_estimation) {
    return false;
  }

  estimateOne(clique_reprojection_features_);

  auto initial_reprojection_error = fillReprojectionErrors_(
      reprojection_estimator_.rot(), 
      reprojection_estimator_.t(),
      clique_reprojection_features_);

  std::sort(
      clique_reprojection_features_.begin(),
      clique_reprojection_features_.end(),
      [](const ReprojectionFeatureWithError& a,
         const ReprojectionFeatureWithError& b) {
        return a.error < b.error;
      });

  // Reestimate using best features from the clique.
  if (clique_reprojection_features_.size() > 5) {
    filtered_reprojection_features_.resize(0);
    for (const auto& f : clique_reprojection_features_) {
       if (f.error < config_.inlier_threshold) {
        filtered_reprojection_features_.push_back(f);
      }
    }
    estimateOne(filtered_reprojection_features_);
  }

  // Reestimate using all good features.
  fillReprojectionErrors_(
      reprojection_estimator_.rot(), 
      reprojection_estimator_.t(),
      all_reprojection_features_);
  filtered_reprojection_features_.resize(0);
  for (const auto& f : all_reprojection_features_) {
    if (f.error < config_.inlier_threshold) {
      filtered_reprojection_features_.push_back(f);
    }
  }

  estimateOne(filtered_reprojection_features_);
  double final_error = fillReprojectionErrors_(
      reprojection_estimator_.rot(), 
      reprojection_estimator_.t(),
      filtered_reprojection_features_);

  std::cout
      << "clique_err=" << initial_reprojection_error
      << " final_err=" << final_error
      << " clique_n=" << clique_reprojection_features_.size()
      << " all_n=" << filtered_reprojection_features_.size()
      << std::endl;

  bool success = final_error < 5.0;
  if (!success) {
    std::cout << "Failure: " << final_error << std::endl;
  }
  return success;
}

void CrossFrameProcessor::estimateOne(
    const std::vector<ReprojectionFeatureWithError>& features) {
  tmp_reprojection_features_.resize(0);

  if (features.size() < 5) {
    std::cout << "WARNING: not enough features" << std::endl;
    return;
  }

  for (auto f : features) {
    tmp_reprojection_features_.push_back(f);
  }

  reprojection_estimator_.estimate(tmp_reprojection_features_);
}
