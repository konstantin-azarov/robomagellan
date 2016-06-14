#ifndef __FRAME_PROCESSOR__HPP__
#define __FRAME_PROCESSOR__HPP__

#include <opencv2/opencv.hpp>
#include <vector>

struct CalibrationData;

class FrameProcessor {
  public:
    FrameProcessor(const CalibrationData& calib);
    
    void process(const cv::Mat src[], cv::Mat& debug);

    void match(const std::vector<cv::KeyPoint>& kps1,
               const std::vector<int>& idx1,
               const cv::Mat& desc1,
               const std::vector<cv::KeyPoint>& kps2,
               const std::vector<int>& idx2,
               const cv::Mat& desc2,
               int inv,
               std::vector<int>& matches);

    void drawDebugImage(cv::Mat& debug);
    
    void printKeypointInfo(int x, int y) const;

    const std::vector<cv::Point3d>& points() const {
      return points_;
    }

    const std::pair<cv::Point2f, cv::Point2f> features(int p) const {
      int i = point_keypoints_[p];
      int j = matches_[0][i];
      return std::make_pair(keypoints_[0][i].pt, keypoints_[1][j].pt);
    }

    // For 0 <= p < points_.size return left and right descriptor for the
    // corresponding point.
    const std::pair<cv::Mat, cv::Mat> pointDescriptors(int p) const {
      int i = point_keypoints_[p];
      int j = matches_[0][i];

      return std::make_pair(descriptors_[0].row(i), descriptors_[1].row(j));
    }

  private:
    const CalibrationData* calib_;

    cv::Mat undistorted_image_[2];

    // [N]: a list of keypoints detected in the left and right image
    std::vector<cv::KeyPoint> keypoints_[2];
    // [NxD]: descriptors corresponding to the keypoints_
    cv::Mat descriptors_[2];              
    // [N] keypoints_[t][order_[t][i]] yields keypoints sorted top-to-bottom, 
    // left-to-right
    std::vector<int> order_[2];               
    // [N] keypoint_[t][i] matches keypoint_[1-t][matches_[t][i]]
    std::vector<int> matches_[2];             
    // points_[match_point_[i]] is extracted from keypoints_[0][i] and 
    // keypoints_[1][matches_[0][i]]
    std::vector<int> match_points_;  
    std::vector<cv::Point3d> points_;
    // point[i] was extracted from keypoints_[0][point_keypoints_[i]]
    std::vector<int> point_keypoints_;
};

#endif
