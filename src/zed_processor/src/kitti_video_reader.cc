#include <boost/format.hpp>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/highgui.hpp>

#include "kitti_video_reader.hpp"

using boost::format;

KittiVideoReader::KittiVideoReader(const std::string& dir) :
    next_frame_id_(0),
    dir_(dir) {
  auto file_name = str(format("%s/image_0/%06d.png") % dir_ % next_frame_id_);
  cv::Mat left = cv::imread(file_name);

  if (left.data == nullptr) {
    std::cout << "File not found: " << file_name << std::endl; 
    abort();
  }

  img_size_ = left.size();
}

KittiVideoReader::~KittiVideoReader() {
}

cv::Size KittiVideoReader::imgSize() const {
  return img_size_;
}

void KittiVideoReader::skip(int cnt) {
  next_frame_id_ += cnt;
}

bool KittiVideoReader::nextFrame(cv::Mat& mat) {
  cv::Mat left = cv::imread(
      str(format("%s/image_0/%06d.png") % dir_ % next_frame_id_), cv::IMREAD_GRAYSCALE);
  cv::Mat right = cv::imread(
      str(format("%s/image_1/%06d.png") % dir_ % next_frame_id_), cv::IMREAD_GRAYSCALE);

  std::cout << "Reading " << str(format("%s/image_0/%06d.png") % dir_ % next_frame_id_) << std::endl;

  if (left.data == nullptr || right.data == nullptr) {
    return false;
  }

  left.copyTo(mat.colRange(0, img_size_.width));
  right.copyTo(mat.colRange(img_size_.width, img_size_.width*2));

  next_frame_id_++;

  return true;
}
