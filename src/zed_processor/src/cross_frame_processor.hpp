#ifndef __CROSS_FRAME_PROCESSOR__HPP__
#define __CROSS_FRAME_PROCESSOR__HPP__

#include <Eigen/Geometry>

#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#include "clique.hpp"
#include "reprojection_estimator.hpp"

#include "frame_processor.hpp"

class StereoCalibrationData;

class FrameData;

struct CrossFrameProcessorConfig {
  int match_radius = 100;
  int x_buckets = 10, y_buckets = 10;
  int max_tracked_features_per_bucket = 20;
  int max_incoming_features = 4000;
  int min_features_for_estimation = 5;
  int max_feature_missing_frames = 1;
  double max_reprojection_error = 2.0;
  double inlier_threshold = 3.0;
  double near_feature_threshold = 20000;

  int maxTrackedFeatures() const {
    return max_tracked_features_per_bucket * x_buckets * y_buckets;
  }
};

struct ReprojectionFeatureWithError : public StereoReprojectionFeature {
  double error;
};

struct WorldFeature {
  Eigen::Vector3d w;
  Eigen::Vector2d left, right;
  int score, age;
  int bucket_id;
  WorldFeature* match;
};

struct CrossFrameDebugData {
  std::vector<WorldFeature> tracked_features, new_features;
  Eigen::Affine3d pose_estimation;
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
    void fillNewFeatures_(const FrameData& d);

    void matchStereo_(
        const cv::Mat_<uint16_t>& scores_l,
        const cv::Mat_<uint16_t>& scores_r);

    void matchMono_(
        const cv::Mat_<uint16_t>& scores,
        std::vector<int>& matches);

    bool isNear_(const WorldFeature& f) const;
    bool isFar_(const WorldFeature& f) const;

    void pickInitialFeatures_();

    void buildCliqueNear_();

    void buildCliqueFar_();

    double deltaL(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2);

    void fillReprojectionFeatures_();

    void updateFeatures_(
        const FrameData& p,
        const Eigen::Quaterniond& r, 
        const Eigen::Vector3d& t);    

    /* double fillReprojectionErrors_( */
    /*     const Eigen::Quaterniond& r, */ 
    /*     const Eigen::Vector3d& tm, */
    /*     std::vector<ReprojectionFeatureWithError>& reprojection_features); */

    bool estimatePose_(
        Eigen::Quaterniond& r, 
        Eigen::Vector3d& t,
        Eigen::Matrix3d* t_cov,
        CrossFrameDebugData* debug_data);

    void estimateOne_(
        const std::vector<StereoReprojectionFeature>& features,
        Eigen::Quaterniond& r, 
        Eigen::Vector3d& t,
        Eigen::Matrix3d* t_cov);

  private:
    CrossFrameProcessorConfig config_;

    const StereoCalibrationData& calibration_;

    Matcher matcher_;

    std::vector<WorldFeature> tracked_features_, new_features_, tmp_features_;

    std::unique_ptr<Stereo<cv::cudev::GpuMat_<uint8_t>>>
      tracked_descriptors_, back_desc_buffer_;

    std::vector<WorldFeature*> features_with_matches_,
      clique_features_,
      near_features_,
      far_features_,
      sorted_features_,
      estimation_features_;
    Stereo<std::vector<const uint8_t*>> new_descriptors_;
    Stereo<CudaDeviceVector<const uint8_t*>> new_descriptors_gpu_;

    Stereo<cv::cudev::GpuMat_<uint16_t>> scores_gpu_;
    Stereo<cv::Mat_<uint16_t>> scores_cpu_;

    // Match from a tracked point to a new point
    Stereo<std::vector<int> > matches_;

     // Clique builder
    Clique clique_;
    // Reprojection features for full_matches_;
    /* std::vector<ReprojectionFeatureWithError> reprojection_features_; */
    /* std::vector<ReprojectionFeatureWithError> filtered_reprojection_features_; */
    std::vector<StereoReprojectionFeature> reprojection_features_;
    // estimators
    StereoReprojectionEstimator reprojection_estimator_;
};

#endif
