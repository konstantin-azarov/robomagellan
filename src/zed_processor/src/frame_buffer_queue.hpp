#ifndef __FRAME_BUFFER__H__
#define __FRAME_BUFFER__H__

#include <stdint.h>
#include <pthread.h>

class FrameBufferQueue {
  public:
    FrameBufferQueue(int frame_size, int length);
    ~FrameBufferQueue();

    void addFrame(const uint8_t* data);
    void nextFrame(uint8_t* data);

  private:
    uint8_t* framePtr_(int index) {
      return data_ + (index * frame_size_);
    }

  private:
    int frame_size_, max_length_;

    uint8_t* data_;
    int read_index_, write_index_, current_size_;

    pthread_mutex_t lock_;
    pthread_cond_t cond_var_;
};

#endif
