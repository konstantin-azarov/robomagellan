#include <iostream>

#include <opencv2/imgproc.hpp>

#include "freak.hpp"

Freak::Freak(double feature_size) : FreakBase(feature_size) {
}

const cv::Mat& Freak::describe(
    const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {
  cv::integral(img, integral_, CV_32S);
    
  int points[kPoints];

  int r = ceil(feature_size_) + 1;
  for (int i = keypoints.size() - 1; i >=0; --i) {
    const auto& kp = keypoints[i];
    if (kp.pt.x <= r || kp.pt.y <= r ||
        kp.pt.x >= img.cols - r ||
        kp.pt.y >= img.rows - r) {
      keypoints.erase(keypoints.begin() + i);
    }
  }
  
  descriptors_.create(keypoints.size(), kPairs/8);
  descriptors_ = 0;

  int i = 0;
  for (auto& kp : keypoints) {
    computePoints_(kp.pt, &patterns_[0], points);

    /* std::cout << "Orientation points:"; */
    /* for (int k = 0; k < kPoints; ++k) { */
    /*   std::cout << " " << points[k]; */
    /* } */
    /* std::cout << std::endl; */

    int dx = 0, dy = 0;
    for (const auto& p : orientation_pairs_) {
      auto d = points[p.i] - points[p.j];
      /* std::cout << p.i << " " << p.j << " " << p.dx << " " << p.dy << " " */ 
      /*   << d << " " << d * p.dx / 2048 << std::endl; */

      dx += d * p.dx / 2048;
      dy += d * p.dy / 2048;
    }

     /* std::cout << "CPU dv: (" << dx << ", " << dy << ")" << std::endl; */ 

    kp.angle = static_cast<float>(atan2((float)dy,(float)dx)*(180.0/CV_PI));

    int orientation = static_cast<int>(kOrientations * kp.angle / 360.0 + 0.5);
    if (orientation < 0) {
      orientation += kOrientations;
    }
    if (orientation >= kOrientations) {
      orientation -= kOrientations;
    }

    /* std::cout << "CPU orientation: " << kp.angle << " " << orientation << std::endl; */

    computePoints_(kp.pt, &patterns_[orientation * kPoints], points);

    /* std::cout << "CPU points:"; */
    /* for (int k = 0; k < kPoints; ++k) { */
    /*   std::cout << " " << points[k]; */
    /* } */
    /* std::cout << std::endl; */

    for (int j=0; j < descriptor_pairs_.size(); ++j) {
      const auto& p = descriptor_pairs_[j];
      int v = points[p.i] >= points[p.j];

      descriptors_[i][j/8] |= v << (j%8);
    }

    ++i;
  }

  return descriptors_;
}

void Freak::computePoints_(
    const cv::Point2f& center, PatternPoint* points, int* res) {
  for (int i = 0; i < kPoints; ++i) {
//    std::cout << points[i].x << " " << points[i].y << std::endl;
    float cx = center.x + points[i].x;
    float cy = center.y + points[i].y;
    float sigma = points[i].sigma;

    int x0 = static_cast<int>(cx - sigma + 0.5);
    int x1 = static_cast<int>(cx + sigma + 1.0 + 0.5);
    int y0 = static_cast<int>(cy - sigma + 0.5);
    int y1 = static_cast<int>(cy + sigma + 1.0 + 0.5);

    int v =   integral_.at<uint32_t>(y1, x1)
            + integral_.at<uint32_t>(y0, x0)
            - integral_.at<uint32_t>(y1, x0) 
            - integral_.at<uint32_t>(y0, x1);

    /* std::cout << "Point " */
    /*   << "(" << center.x << ", " << center.y << ") " << i << " -> " */
    /*   << "(" << x0 << ", " << y0 << ") - (" << x1 << ", " << y1 << ") = " */
    /*   << v << std::endl; */

    res[i] = static_cast<int>(v / ((x1 - x0) * (y1 - y0)));
  }
}


