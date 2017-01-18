#ifndef __CROSS_FRAME_PROCESSOR__HPP__
#define __CROSS_FRAME_PROCESSOR__HPP__

#include <Eigen/Geometry>

#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#include "clique.hpp"
#include "reprojection_estimator.hpp"
#include "rigid_estimator.hpp"

#include "frame_processor.hpp"

class StereoCalibrationData;

class FrameData;

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
  
  WorldFeature *match;
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

    Matcher matcher_;

    std::unique_ptr<Bucketizer<WorldFeature>> tracked_features_;
    // Only one index, because descriptors are compacted in the end
    std::vector<WorldFeature*> feature_index_;
    std::unique_ptr<Stereo<DescriptorBuffer>>
      tracked_descriptors_, back_desc_buffer_;

    std::unique_ptr<Bucketizer<WorldFeature>> new_features_;
    Stereo<std::vector<WorldFeature*>> new_feature_index_;

    Stereo<std::vector<ushort2>> feature_pairs_;


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
