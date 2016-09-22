#ifndef __CROSS_FRAME_PROCESSOR__HPP__
#define __CROSS_FRAME_PROCESSOR__HPP__

#include <vector>

#include "clique.hpp"
#include "reprojection_estimator.hpp"
#include "rigid_estimator.hpp"

class StereoCalibrationData;

class FrameProcessor;

struct CrossFrameMatch {
  CrossFrameMatch() {}

  CrossFrameMatch(
      const cv::Point3d& p1_, 
      const cv::Point3d& p2_, 
      int bucket_index_,
      float score_,
      int i1_, 
      int i2_) :
    p1(p1_), p2(p2_), bucket_index(bucket_index_), score(score_), i1(i1_), i2(i2_) {
  }

  cv::Point3d p1, p2;
  // Bucketing
  int bucket_index;
  float score;
  // Index of the corresponding frame point
  int i1, i2;
};

struct CrossFrameProcessorConfig {
  int match_radius = 100;
  int x_buckets = 10, y_buckets = 10;
  int max_features_per_bucket = 5;
  int min_features_for_estimation = 5;
  double max_reprojection_error = 2.0;
  double inlier_threshold = 3.0;
};

struct ReprojectionFeatureWithError : public ReprojectionFeature {
  double error;
};

class CrossFrameProcessor {
  public:
    CrossFrameProcessor(
        const StereoCalibrationData& calibration,
        const CrossFrameProcessorConfig& config);
    
    bool process(const FrameProcessor& p1, const FrameProcessor& p2);
    
    const cv::Mat& rot() const { return reprojection_estimator_.rot(); }
    const cv::Point3d& t() const { return reprojection_estimator_.t(); } 
    const cv::Mat& t_cov() const { return reprojection_estimator_.t_cov(); }

    const std::vector<CrossFrameMatch>& fullMatches() const { 
      return full_matches_; 
    }

    const std::vector<int>& filteredMatches() const {
      return filtered_matches_;
    }

    const std::vector<int>& clique() const {
      return clique_.clique();
    }

    const std::vector<ReprojectionFeatureWithError>& cliqueFeatures() const { 
      return clique_reprojection_features_;
    }

    const std::vector<ReprojectionFeatureWithError>& inlierFeatures() const {
      return filtered_reprojection_features_;
    }

    const StereoCalibrationData& calibration() const { 
      return calibration_;
    }

/*    const std::vector<ReprojectionFeature>& reprojectionFeatures() const {
      return reprojection_features_;
    }*/
  private:
    void match(
        const FrameProcessor& p1, 
        const FrameProcessor& p2,
        std::vector<int>& matches);

    void buildClique_(
        const FrameProcessor& p1, 
        const FrameProcessor& p2);

    double deltaL(
        const cv::Point3d& p1,
        const cv::Point3d& p2);

    void fillReprojectionFeatures_(
        const FrameProcessor& p1,
        const FrameProcessor& p2);

    double fillReprojectionErrors_(
        const cv::Mat& R, 
        const cv::Point3d& tm,
        std::vector<ReprojectionFeatureWithError>& reprojection_features);

    bool estimatePose();

    void estimateOne(const std::vector<ReprojectionFeatureWithError>& features);

  private:
    CrossFrameProcessorConfig config_;

    const StereoCalibrationData& calibration_;

    // matches[0][i] - best match in the second frame for i-th feature in the first frame
    // matches[1][j] - best match in the first frame for j-th feature in the second frame
    std::vector<int> matches_[2];
    // 3d point matches between frames
    std::vector<CrossFrameMatch> full_matches_;
    std::vector<int> filtered_matches_;
    // Clique builder
    Clique clique_;
    // Reprojection features for full_matches_;
    std::vector<ReprojectionFeatureWithError> all_reprojection_features_;
    std::vector<ReprojectionFeatureWithError> clique_reprojection_features_;
    std::vector<ReprojectionFeatureWithError> filtered_reprojection_features_;
    std::vector<ReprojectionFeature> tmp_reprojection_features_;
    // estimators
    RigidEstimator rigid_estimator_;
    ReprojectionEstimator reprojection_estimator_;
};

#endif
