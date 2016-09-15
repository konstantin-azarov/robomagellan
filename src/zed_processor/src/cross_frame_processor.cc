#include "cross_frame_processor.hpp"
#include "frame_processor.hpp"
#include "math3d.hpp"

const double CROSS_POINT_DIST_THRESHOLD = 20; // mm

CrossFrameProcessor::CrossFrameProcessor(const StereoIntrinsics* intrinsics) 
  : intrinsics_(*intrinsics), reprojection_estimator_(intrinsics) {
}

bool CrossFrameProcessor::process(
    const FrameProcessor& p1, 
    const FrameProcessor& p2) {
  const std::vector<cv::Point3d>& points1 = p1.points();
  const std::vector<cv::Point3d>& points2 = p2.points();

#ifdef DEBUG
  std::cout << "Points 1" << std::endl;
  for (int i=0; i < points1.size(); ++ i) {
    std::cout << points1[i] << std::endl;
  }
  std::cout << "Points 2" << std::endl;
  for (int i=0; i < points2.size(); ++ i) {
    std::cout << points2[i] << std::endl;
  }
//
//  std::cout << "kp1" << p1.keypoints(0)[p1.pointKeypoints()[326]] << std::endl;
//  std::cout << "kp2" << p2.keypoints(0)[p2.pointKeypoints()[326]] << std::endl;
#endif

  match(p1, p2, matches_[0]);
  match(p2, p1, matches_[1]);

  full_matches_.resize(0);

  for (int i=0; i < matches_[0].size(); ++i) {
    int j = matches_[0][i];
    if (j != -1 && matches_[1][j] == i) {
      full_matches_.push_back(CrossFrameMatch(points1[i], points2[j], i, j));
    }
  }

#ifdef DEBUG
  std::cout << "Full matches (" << full_matches_.size() << ")" << std::endl;
  for (int i=0; i < full_matches_.size(); ++i) {
    const auto& m = full_matches_[i];
    std::cout << m.p1 << " " << m.p2 
      << " " << m.i1 << " " << m.i2 << " " 
//      << " " << p1.pointDescriptors(m.i1).first << p2.pointDescriptors(m.i2).first 
//      << p1.keypoints(0)[p1.pointKeypoints()[m.i1]] << " "
 //     << p2.keypoints(0)[p2.pointKeypoints()[m.i2]] << " "
      << std::endl;
  }
#endif

  buildClique_(p1, p2);

  std::cout << "Matches = " << full_matches_.size() << "; clique = " << clique_points_[0].size() << std::endl;

  if (clique_points_[0].size() < 10) {
    return false;
  }

  rigid_estimator_.estimate(clique_points_[0], clique_points_[1]);
  reprojection_estimator_.estimate(reprojection_features_);

//  auto rigid_errors = reprojectionError_(
//      rigid_estimator_.rot(), rigid_estimator_.t());
//  std::cout << "Reprojection errors rigid ([1->2] [2->1]):" 
//    << rigid_errors.first << " " << rigid_errors.second  << std::endl;

  /* auto reprojection_errors = reprojectionError_( */
  /*     reprojection_estimator_.rot(), reprojection_estimator_.t()); */
  /* std::cout << "Reprojection errors 1 ([1->2] [2->1]):" */ 
  /*   << reprojection_errors.first << " " << reprojection_errors.second  << std::endl; */

  reprojection_estimator_.estimate(reprojection_features_);

  /* reprojection_errors = reprojectionError_( */
  /*     reprojection_estimator_.rot(), reprojection_estimator_.t()); */
  /* std::cout << "Reprojection errors 2 ([1->2] [2->1]):" */ 
  /*   << reprojection_errors.first << " " << reprojection_errors.second  << std::endl; */

  /* std::cout << "dR = " << rot() << "; dt = " << t() << std::endl; */

  return true;
}

void CrossFrameProcessor::match(
    const FrameProcessor& p1, 
    const FrameProcessor& p2,
    std::vector<int>& matches) {

  const std::vector<cv::Point3d>& points1 = p1.points();
  const std::vector<cv::Point3d>& points2 = p2.points();

  matches.resize(points1.size());

  for (int i = 0; i < points1.size(); ++i) {
    int best_j = -1;
    double best_dist = 1E+15;
    auto left_descs = p1.pointDescriptors(i);

//    if (i == 326) {
//      std::cout << "Matching 326" << std::endl;
//    }
//
    for (int j = 0; j < points2.size(); ++j) {
      auto right_descs = p2.pointDescriptors(j);

      double d1 = descriptorDist(left_descs.first, right_descs.first);
      double d2 = descriptorDist(left_descs.second, right_descs.second);

      double d = std::max(d1, d2);

//      if (i == 326) {
//        std::cout << j << "=" << d << std::endl;
//      }

      if (d < best_dist) {
        best_dist = d;
        best_j = j;
      }
    }

    matches[i] = best_j;
  }
}

void CrossFrameProcessor::buildClique_(
    const FrameProcessor& p1, 
    const FrameProcessor& p2) {
  // Construct the matrix of the matches
  // matches i and j have an edge is distance between corresponding points in the
  // first image is equal to the distance between corresponding points in the
  // last image.
  int n = full_matches_.size();
  clique_.reset(n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      auto& m1 = full_matches_[i];
      auto& m2 = full_matches_[j];
      double d1 = sqrt(norm3(m1.p1 - m2.p1));
      double d2 = sqrt(norm3(m1.p2 - m2.p2));
     
//      std::cout << i << " - " << j << ": " << d1 << ", " << d2  << std::endl;

      if (std::abs(d1 - d2) < CROSS_POINT_DIST_THRESHOLD) {
//        std::cout << "Edge: " << i << " <-> " << j << std::endl;
        clique_.addEdge(i, j);
      }
    }
  }

  const std::vector<int>& clique = clique_.clique();

  clique_points_[0].resize(clique.size());
  clique_points_[1].resize(clique.size());
  reprojection_features_.resize(clique.size());
  for (int i=0; i < clique.size(); ++i) {
    const auto& m = full_matches_[clique[i]];

    clique_points_[0][i] = m.p1;
    clique_points_[1][i] = m.p2;

    auto& f = reprojection_features_[i];

    f.r1 = m.p1;
    f.r2 = m.p2;
    std::tie(f.s1l, f.s1r) = p1.features(m.i1);
    std::tie(f.s2l, f.s2r) = p2.features(m.i2);
  }
}

bool cmp(
    std::pair<double, ReprojectionFeature> a, 
    std::pair<double, ReprojectionFeature> b) {
  return a.first < b.first;
}

std::pair<double, double> CrossFrameProcessor::reprojectionError_(
    const cv::Mat& R, 
    const cv::Point3d& tm) {
  double res1 = 0, res2 = 0;

  std::vector<std::pair<double, ReprojectionFeature>> features_with_errors;

  std::cout << "Reprojection errors: " << std::endl;
  for (auto f : reprojection_features_) {
    auto p1 = projectPoint(intrinsics_, R.t()*(f.r2 - tm));
    auto p2 = projectPoint(intrinsics_, R*f.r1 + tm);

    double e1 = norm2(p1.first - f.s1l) + norm2(p1.second - f.s1r); 
    double e2 = norm2(p2.first - f.s2l) + norm2(p2.second - f.s2r);

    std::cout << e1 + e2 << " ";

    features_with_errors.push_back(std::make_pair(e1 + e2, f));

    res1 += e1;
    res2 += e2; 
  }
  std::cout << std::endl;


  int n = reprojection_features_.size();
  
  std::sort(features_with_errors.begin(), features_with_errors.end(), cmp);

  reprojection_features_.resize(0);
  for (int i=0; i < 7 && i < features_with_errors.size(); ++i) {
    reprojection_features_.push_back(features_with_errors[i].second);
  }

  return std::make_pair(res1 / n, res2 / n);
}

