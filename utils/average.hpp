#ifndef __AVERAGE_HPP__
#define __AVERAGE_HPP__

class Average {
  public: 
    Average(int avg_samples) : 
      avg_samples_(avg_samples), 
      counter_(0),
      current_sum_(0), 
      current_avg_(0) {}

    void sample(double v) {
      current_sum_ += v;
      counter_++;
      if (counter_ == avg_samples_) {
        current_avg_ = current_sum_ / avg_samples_;
        counter_ = 0;
        current_sum_ = 0;
      }
    }

    double value() const { 
      return current_avg_;
    }

  private:
    int avg_samples_, counter_;
    double current_sum_, current_avg_;

};

#endif
