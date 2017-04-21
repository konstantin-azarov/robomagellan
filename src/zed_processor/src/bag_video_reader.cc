#include <assert.h>

#include <opencv2/imgproc.hpp>

#include <rosbag/bag.h>
#include <rosbag/query.h>
#include <rosbag/view.h>
#include <ros/console.h>

#include "bag_video_reader.hpp"

BagVideoReader::BagVideoReader(
    const std::string& filename, 
    const std::string& topic) :
    bag_(filename, rosbag::bagmode::Read),
    view_(bag_, rosbag::TopicQuery(topic)),
    iterator_(view_.begin()) {
}

BagVideoReader::~BagVideoReader() {
}

void BagVideoReader::skip(int cnt) {
  if (cnt > 0) {
    while (iterator_ != view_.end() && cnt != 0) {
      iterator_++;
      cnt--;
    }
  } else {
    assert(false);
  }
}

bool BagVideoReader::nextFrame(cv::Mat& mat) {
  if (iterator_ == view_.end())
    return false;

  auto img = iterator_->instantiate<sensor_msgs::Image>();
  cv::Mat img_mat(
      img->height, 
      img->width, 
      CV_8UC3, 
      img->data.data(), 
      img->step);

  ++iterator_;

  if (img->encoding == "bgr8") {
    img_mat.copyTo(mat);
    return true;
  } else {
    ROS_ERROR_STREAM("Unknown image format: " << img->encoding);
    return false;
  }
}

bool BagVideoReader::nextFrameRaw(cv::Mat& mat) {
  if (iterator_ == view_.end())
    return false;

  auto img = iterator_->instantiate<sensor_msgs::Image>();
  cv::Mat img_mat(
      img->height, 
      img->width, 
      CV_8UC3, 
      img->data.data(), 
      img->step);

  img_mat.copyTo(mat);
  
  ++iterator_;

  return true;
}

