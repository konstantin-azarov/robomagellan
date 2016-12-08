#ifndef __KITTI_VIDEO_READER__
#define __KITTI_VIDEO_READER__

#include <string>

#include <opencv2/core.hpp>

#include "video_reader.hpp"

class KittiVideoReader : public VideoReader {
  public:
    KittiVideoReader(const std::string& dir);
    virtual ~KittiVideoReader();

    void skip(int cnt);

    bool nextFrame(cv::Mat& mat);

    cv::Size imgSize() const;

  private:
    int next_frame_id_;
    std::string dir_;
    cv::Size img_size_;
};

#endif
