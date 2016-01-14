#include <pthread.h>
#include <cstring>
#include <iostream>

#include "frame_buffer_queue.hpp"

using namespace std;

FrameBufferQueue::FrameBufferQueue(int frame_size, int max_length) {
  frame_size_ = frame_size;
  max_length_ = max_length;

  read_index_ = 0;
  write_index_ = 0;
  current_size_ = 0;

  data_ = new uint8_t[max_length*frame_size_];

  pthread_mutex_init(&lock_, 0);
  pthread_cond_init(&cond_var_, 0);
}

FrameBufferQueue::~FrameBufferQueue() {
  delete[] data_;
  pthread_mutex_destroy(&lock_);
  pthread_cond_destroy(&cond_var_);
}

void FrameBufferQueue::addFrame(const uint8_t* data) {
  pthread_mutex_lock(&lock_);
  if (current_size_ < max_length_) {
    memcpy(
      framePtr_(write_index_),
      data,
      frame_size_);

    write_index_ = (write_index_ + 1) % max_length_;
    current_size_++;
    pthread_cond_signal(&cond_var_);
  } else {
    cerr << "Queue overflow" << endl;
  }
  pthread_mutex_unlock(&lock_);
}

void FrameBufferQueue::nextFrame(uint8_t* data) {
  pthread_mutex_lock(&lock_);
  while (current_size_ == 0) {
    pthread_cond_wait(&cond_var_, &lock_);
  }
  memcpy(
    data,
    framePtr_(read_index_),
    frame_size_);
  read_index_ = (read_index_ + 1) % max_length_;
  current_size_--;
  pthread_mutex_unlock(&lock_);
}
