#ifndef __FPS_METER__HPP__
#define __FPS_METER__HPP__

#include "utils.hpp"

class FpsMeter {
  public:
    FpsMeter(int frames) : 
      frame_counter_(0), 
      frames_to_count_(frames), 
      t0_(nanoTime()), 
      current_fps_(0) {};

    void mark() {
      frame_counter_++;
      if (frame_counter_ == frames_to_count_) {
        auto t = nanoTime();
        current_fps_ = frames_to_count_ / (t - t0_);
        t0_ = t;
        frame_counter_ = 0;
      }
    }

    double currentFps() const {
      return current_fps_;
    }

  private:
    int frame_counter_, frames_to_count_;
    double t0_, current_fps_;
};

#endif
