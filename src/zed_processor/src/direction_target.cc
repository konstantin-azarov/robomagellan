#include "direction_target.hpp"

DirectionTarget DirectionTarget::read(const std::string& filename) {
  DirectionTarget target;

  cv::FileStorage fs(filename, cv::FileStorage::READ);

  fs["image"] >> target.image_file;
  fs["target"] >> target.target;
 
  int n = fs["features"].size();
  target.descriptors.create(n, 512/8, CV_8UC1);
  target.keypoints.resize(n);

  int i = 0;
  for (const auto& feature_s : fs["features"]) {
    cv::Point2d kp;
    feature_s["pt"] >> target.keypoints[i];
    cv::Mat desc;
    feature_s["desc"] >> desc;
    desc.copyTo(target.descriptors.row(i));
    ++i;
  }

  return target;
}

void DirectionTarget::write(const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  fs << "image" << image_file;
  fs << "target" << target;
  fs << "features" << "[";
  for (int i = 0; i < keypoints.size(); ++i) {
    fs << "{" << "pt" << keypoints[i] << "desc" << descriptors.row(i) << "}";
  }
  fs << "]";
}
