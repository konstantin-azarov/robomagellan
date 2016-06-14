#ifndef __MATH_3D__HPP__
#define __MATH_3D__HPP__

#include <assert.h>
#include <opencv2/opencv.hpp>

#include "calibration_data.hpp"

cv::Point3d operator * (const cv::Mat& m, const cv::Point3d& p);

cv::Point2f projectPoint(const cv::Mat& m, const cv::Point3d& p);

std::pair<cv::Point2d, cv::Point2d> projectPoint(
    const StereoIntrinsics& i, 
    const cv::Point3d& p);

cv::Mat rotX(double angle);

cv::Mat rotY(double angle);

cv::Mat rotZ(double angle);

bool compareMats(const cv::Mat& l, const cv::Mat& r, double eps);

bool comparePoints(const cv::Point3d& l, const cv::Point3d& r, double eps);

double norm2(const cv::Point2f& pt);

double norm3(const cv::Point3d& pt);

double descriptorDist(const cv::Mat& a, const cv::Mat& b);

#endif
