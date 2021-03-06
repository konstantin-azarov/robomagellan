#include "raw_video_reader.hpp"

using namespace std;

RawVideoReader::RawVideoReader(const string& filename, int w, int h) : w_(w), h_(h) {
  f_.open(filename, ios::in | ios::binary);
}

RawVideoReader::~RawVideoReader() {
}

bool RawVideoReader::nextFrame(uint8_t* data) {
  if (!f_.eof()) {
    f_.read(reinterpret_cast<char*>(data), w_*h_);
    return true;
  } else {
    return false;
  }
}

void RawVideoReader::skip(int frames) {
  f_.seekg(frames*w_*h_, ios_base::cur);
}
