#ifndef __VIDEO_READER__HPP__
#define __VIDEO_READER__HPP__

#include <opencv2/core.hpp>

class VideoReader {
  public:
    virtual ~VideoReader() {};

    virtual void skip(int cnt) = 0;

    virtual bool nextFrame(cv::Mat& mat) = 0;
};

#endif
