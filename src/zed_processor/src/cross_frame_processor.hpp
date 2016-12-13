#ifndef __CROSS_FRAME_PROCESSOR__HPP__
#define __CROSS_FRAME_PROCESSOR__HPP__

#include <Eigen/Geometry>

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#include "clique.hpp"
#include "reprojection_estimator.hpp"
#include "rigid_estimator.hpp"

class StereoCalibrationData;

class FrameData;

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

struct ReprojectionFeatureWithError : public StereoReprojectionFeature {
  double error;
};

struct CrossFrameDebugData {
  std::vector<ReprojectionFeatureWithError> all_reprojection_features;
  std::vector<std::vector<ReprojectionFeatureWithError> > reprojection_features;
  std::vector<Eigen::Affine3d> pose_estimations;
};

template <class T>
class Bucketizer {
  public:
    Bucketizer(std::vector<int> bucket_sizes) {
      buckets_.resize(bucket_sizes.size());
      for (int i=0; i < bucket_sizes.size(); ++i) {
        buckets_[i].reserve(bucket_sizes[i]);
      }
    }
  
    void clear() {
      for (auto& b : buckets_) {
        b.resize(0);
      }
    }

    template <class F>
    void bucketize(const std::vector<T>& items, F classifier) {
      for (const auto& i : items) {
        int v = classifier(i);
        if (v != -1) {
          auto& b = buckets_[v];
          if (b.size() < bucket_size_limit_[v]) {
            b.push_back(i);
          }
        }
      }
    }

    const std::vector<T>& bucket(int i) {
      return buckets_[i];
    }

    template <class F>
    void sortAll(F f) {
      for (auto& b : buckets_) {
        std::sort(std::begin(b), std::end(b), f);
      }
    }

  private:
    std::vector<int> bucket_size_limit_;
    std::vector<std::vector<T> > buckets_;
};

struct WorldFeature {
  Eigen::Vector3d w;
  Eigen::Vector2i left, right;
  int desc_l, desc_r;
  int score;
};

class CrossFrameProcessor {
  public:
    CrossFrameProcessor(
        const StereoCalibrationData& calibration,
        const CrossFrameProcessorConfig& config);
    
    bool process(
        const FrameData& p1, const FrameData& p2,
        bool use_initial_estimate,
        Eigen::Quaterniond& r, 
        Eigen::Vector3d& t,
        Eigen::Matrix3d* t_cov,
        CrossFrameDebugData* debug_data);
   
  private:
    void match(
        const FrameData& p1, 
        const FrameData& p2,
        const cv::Mat_<ushort>& scores,
        int n1, int n2,
        std::vector<int>& matches);

    void buildCliqueNear_(
        const FrameData& p1, 
        const FrameData& p2);
    void buildCliqueFar_(
        const FrameData& p1, 
        const FrameData& p2);

    double deltaL(
        const cv::Point3d& p1,
        const cv::Point3d& p2);

    void fillReprojectionFeatures_(
        const FrameData& p1,
        const FrameData& p2);

    double fillReprojectionErrors_(
        const Eigen::Quaterniond& r, 
        const Eigen::Vector3d& tm,
        std::vector<ReprojectionFeatureWithError>& reprojection_features);

    bool estimatePose_(
        bool use_initial_estimate,
        Eigen::Quaterniond& r, 
        Eigen::Vector3d& t,
        Eigen::Matrix3d* t_cov,
        CrossFrameDebugData* debug_data);

    void estimateOne_(
        const std::vector<ReprojectionFeatureWithError>& features,
        Eigen::Quaterniond& r, 
        Eigen::Vector3d& t,
        Eigen::Matrix3d* t_cov);

  private:
    CrossFrameProcessorConfig config_;

    const StereoCalibrationData& calibration_;

    cv::cudev::GpuMat_<ushort> scores_left_gpu_, scores_right_gpu_;
    cv::Mat_<ushort> scores_left_, scores_right_;

    // matches[0][i] - best match in the second frame for i-th feature in the first frame
    // matches[1][j] - best match in the first frame for j-th feature in the second frame
    std::vector<int> matches_left_[2], matches_right_[2];
    // 3d point matches between frames
    std::vector<CrossFrameMatch> full_matches_;
    std::vector<int> filtered_matches_;
    // Clique builder
    Clique clique_;
    // Reprojection features for full_matches_;
    std::vector<ReprojectionFeatureWithError> all_reprojection_features_;
    std::vector<ReprojectionFeatureWithError> filtered_reprojection_features_;
    std::vector<StereoReprojectionFeature> tmp_reprojection_features_;
    // estimators
    RigidEstimator rigid_estimator_;
    StereoReprojectionEstimator reprojection_estimator_;
};

#endif
