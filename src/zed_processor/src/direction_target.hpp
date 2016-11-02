#ifndef __DIRECTION_TARGET__HPP__
#define __DIRECTION_TARGET__HPP__

#include <opencv2/core.hpp>
#include <vector>

struct DirectionTarget {
  static DirectionTarget read(const std::string& filename);
  void write(const std::string& filename);

  std::string image_file;

  cv::Point2d target;

  std::vector<cv::Point2d> keypoints;
  cv::Mat descriptors;
};

#endif
