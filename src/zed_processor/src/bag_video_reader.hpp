#ifndef __BAG_VIDEO_READER__HPP__
#define __BAG_VIDEO_READER__HPP__

#include <string>
#include <memory>

#include <opencv2/core.hpp>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include "sensor_msgs/Image.h"

namespace rosbag {
  class Bag;
}

class BagVideoReader {
  public:
    BagVideoReader(
        const std::string& filename, 
        const std::string& topic);

    ~BagVideoReader();

    void skip(int cnt);

    bool nextFrame(cv::Mat& mat);

  private:
    rosbag::Bag bag_;
    rosbag::View view_;
    rosbag::View::iterator iterator_;
};

#endif
