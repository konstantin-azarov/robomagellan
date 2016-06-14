#ifndef __TIMER__HPP__
#define __TIMER__HPP__

#include "utils.hpp"

class Timer {
  public:
    Timer() : t0_(nanoTime()) {}

    double mark() { 
      auto t = nanoTime();
      auto res = t - t0_;
      t0_ = t;
      return res;
    }; 

  private:
    double t0_;
};

#endif
