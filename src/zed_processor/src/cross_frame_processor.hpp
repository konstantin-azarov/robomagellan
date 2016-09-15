#ifndef __CROSS_FRAME_PROCESSOR__HPP__
#define __CROSS_FRAME_PROCESSOR__HPP__

#include <vector>

#include "clique.hpp"
#include "reprojection_estimator.hpp"
#include "rigid_estimator.hpp"

struct StereoIntrinsics;

class FrameProcessor;

struct CrossFrameMatch {
  CrossFrameMatch() {}

  CrossFrameMatch(const cv::Point3d& p1_, const cv::Point3d& p2_, int i1_, int i2_) :
    p1(p1_), p2(p2_), i1(i1_), i2(i2_) {
  }

  cv::Point3d p1, p2;
  // Index of the corresponding frame point
  int i1, i2;
};

class CrossFrameProcessor {
  public:
    CrossFrameProcessor(const StereoIntrinsics* intrinsics);
    
    bool process(const FrameProcessor& p1, const FrameProcessor& p2);
    
    const cv::Mat& rot() const { return reprojection_estimator_.rot(); }
    const cv::Point3d& t() const { return reprojection_estimator_.t(); } 

    const std::vector<CrossFrameMatch>& fullMatches() const { 
      return full_matches_; 
    }

    const std::vector<ReprojectionFeature>& reprojectionFeatures() const {
      return reprojection_features_;
    }
  private:
    void match(
        const FrameProcessor& p1, 
        const FrameProcessor& p2,
        std::vector<int>& matches);

    void buildClique_(
        const FrameProcessor& p1, 
        const FrameProcessor& p2);

    std::pair<double, double> reprojectionError_(
        const cv::Mat& R, 
        const cv::Point3d& tm);

  private:
    const StereoIntrinsics& intrinsics_;

    // matches[0][i] - best match in the second frame for i-th feature in the first frame
    // matches[1][j] - best match in the first frame for j-th feature in the second frame
    std::vector<int> matches_[2];
    // 3d point matches between frames
    std::vector<CrossFrameMatch> full_matches_;
    // Clique builder
    Clique clique_;
    // Points corresponding to clique from the first and second frames
    std::vector<cv::Point3d> clique_points_[2];
    // Original features (or rather their locations) from the first and second frames
    // respectively
    std::vector<ReprojectionFeature> reprojection_features_;
    // estimators
    RigidEstimator rigid_estimator_;
    ReprojectionEstimator reprojection_estimator_;
};

#endif
