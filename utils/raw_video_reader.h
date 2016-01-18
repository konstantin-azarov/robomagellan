#ifndef __RAW_VIDEO_READER_H__
#define __RAW_VIDEO_READER_H__

#include <fstream>
#include <string>

class RawVideoReader {
  public:
    RawVideoReader(const std::string& filename, int w, int h);
    ~RawVideoReader();

    bool nextFrame(uint8_t* data);

  private:
    std::ifstream f_;
    int w_, h_;
};

#endif
