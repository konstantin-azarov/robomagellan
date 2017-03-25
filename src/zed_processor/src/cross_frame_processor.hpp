#ifndef __CROSS_FRAME_PROCESSOR__HPP__
#define __CROSS_FRAME_PROCESSOR__HPP__

#include <Eigen/Geometry>

#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#include "bucketizer.hpp"
#include "clique.hpp"
#include "reprojection_estimator.hpp"
#include "rigid_estimator.hpp"

#include "frame_processor.hpp"

class StereoCalibrationData;

class FrameData;

struct CrossFrameProcessorConfig {
  int match_radius = 100;
  int x_buckets = 10, y_buckets = 10;
  int max_tracked_features_per_bucket = 20;
  int max_incoming_features = 4000;
  int min_features_for_estimation = 5;
  double max_reprojection_error = 2.0;
  double inlier_threshold = 3.0;

  int maxTrackedFeatures() const {
    return max_tracked_features_per_bucket * x_buckets * y_buckets;
  }
};

struct ReprojectionFeatureWithError : public StereoReprojectionFeature {
  double error;
};

struct CrossFrameDebugData {
  std::vector<ReprojectionFeatureWithError> all_reprojection_features;
  std::vector<std::vector<ReprojectionFeatureWithError> > reprojection_features;
  std::vector<Eigen::Affine3d> pose_estimations;
};

struct WorldFeature {
  Eigen::Vector3d w;
  Eigen::Vector2i left, right;
  int desc_l, desc_r;
  int score, age;
  WorldFeature* match;
};

class CrossFrameProcessor {
  public:
    CrossFrameProcessor(
        const StereoCalibrationData& calibration,
        const CrossFrameProcessorConfig& config);
    
    bool process(
        const FrameData& p,
        Eigen::Quaterniond& r, 
        Eigen::Vector3d& t,
        Eigen::Matrix3d* t_cov,
        CrossFrameDebugData* debug_data);
   
  private:
    void matchStereo_();

    void matchMono_(
        const cv::Mat_<uint16_t>& scores,
        std::vector<int>& matches);

    bool isNear(const WorldFeature& f) const;
    bool isFar(const WorldFeature& f) const;

    void pickInitialFeatures_();

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

    std::vector<WorldFeature> tracked_features_;
    std::unique_ptr<Stereo<cv::cudev::GpuMat_<uint8_t>>>
      tracked_descriptors_, back_desc_buffer_;

    std::vector<WorldFeature*> features_with_matches_;
    std::vector<WorldFeature*> near_features_, far_features_;

    std::vector<WorldFeature*> estimation_features_;

    std::vector<WorldFeature> new_features_;

    Stereo<cv::cudev::GpuMat_<uint16_t>> scores_gpu_;
    Stereo<cv::Mat_<uint16_t>> scores_cpu_;

    // Match from a tracked point to a new point
    Stereo<std::vector<int> > matches_;

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
