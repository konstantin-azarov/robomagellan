#ifndef __MATH_3D__HPP__
#define __MATH_3D__HPP__

#include <assert.h>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calibration_data.hpp"

cv::Point3d operator * (const cv::Mat& m, const cv::Point3d& p);

cv::Point2f projectPoint(const cv::Mat& m, const cv::Point3d& p);

std::pair<Eigen::Vector2d, Eigen::Vector2d> projectPoint(
    const StereoIntrinsics& camera, 
    const Eigen::Vector3d& p);

cv::Point2d projectPoint(const MonoIntrinsics& camera, const cv::Point3d& p);

/**
 * Convert a quaternion (from current coordinate system to reference 
 * coordinate system) to yaw/pitch/roll (in this order around local axes).
 */
Eigen::Vector3d rotToYawPitchRoll(const Eigen::Quaterniond& r);

bool compareMats(const cv::Mat& l, const cv::Mat& r, double eps);

bool comparePoints(const cv::Point3d& l, const cv::Point3d& r, double eps);

double norm2(const cv::Point2f& pt);

double norm3(const cv::Point3d& pt);

double descriptorDist(const cv::Mat& a, const cv::Mat& b);

cv::Mat hconcat(const cv::Mat& l, const cv::Mat& r);

cv::Mat leastSquares(const cv::Mat& x, const cv::Mat& y);

cv::Vec3d fitLine(const cv::Mat& pts);

cv::Vec2d intersectLines(const cv::Mat& lines);


#endif
