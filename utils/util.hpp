#include <assert.h>
#include <opencv2/opencv.hpp>

inline cv::Point3d operator * (const cv::Mat& m, const cv::Point3d& p) {
  assert(m.rows == 3 && m.cols == 3);

  const cv::Mat_<double>& r = static_cast<const cv::Mat_<double>&>(m);

  return cv::Point3d(r(0, 0) * p.x + r(0, 1) * p.y + r(0, 2) * p.z,
                     r(1, 0) * p.x + r(1, 1) * p.y + r(1, 2) * p.z,
                     r(2, 0) * p.x + r(2, 1) * p.y + r(2, 2) * p.z);
}

inline cv::Point2f projectPoint(const cv::Mat& m, const cv::Point3d& p) {
  assert(m.rows == 3 && m.cols == 4);
  
  const cv::Mat_<double>& P = static_cast<const cv::Mat_<double>&>(m);

  return cv::Point2f((P(0, 0) * p.x + P(0, 2) * p.z + P(0, 3))/p.z,
                     (P(1, 1) * p.y + P(1, 2) * p.z)/p.z);
}

inline double norm2(const cv::Point2f& pt) {
  return pt.x*pt.x + pt.y*pt.y;
}
