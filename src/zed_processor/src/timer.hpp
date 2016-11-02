#ifndef __TIMER__HPP__
#define __TIMER__HPP__

#include <chrono>
#include <sstream>
#include <vector>

class Timer {
  public:
    Timer() : t0_(std::chrono::high_resolution_clock::now()) {}

    void mark(const char* name) { 
      auto t = std::chrono::high_resolution_clock::now();
      log_.push_back(std::make_pair(name, t - t0_));
      t0_ = t;
    }; 

    void advance() {
      t0_ = std::chrono::high_resolution_clock::now();
    }

    std::string str() {
      std::ostringstream os;

      bool sep = false;
      for (const auto& e : log_) {
        if (sep) {
          os << "; ";
        } else {
          sep = true;
        }

        os << e.first << "=" 
          << std::chrono::duration_cast<std::chrono::milliseconds>(e.second).count();
      }

      return os.str();
    }

  private:
    std::chrono::high_resolution_clock::time_point t0_;
    std::vector<
      std::pair<
        const char*,
        std::chrono::high_resolution_clock::duration>> log_;
};

#endif
