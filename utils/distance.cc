#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

#define BACKWARD_HAS_DW 1

#include "backward.hpp"
#include "calibration_data.hpp"

namespace po = boost::program_options;

const int frame_width = 640;
const int frame_height = 480;

backward::SignalHandling sh;

int main(int argc, char** argv) {
  std::string calib_file, image_file;

  po::options_description desc("Command line options");
  desc.add_options()
      ("calib-file",
       po::value(&calib_file)->default_value("data/calib.yml"),
       "path to the calibration file");
  
  desc.add_options()
      ("image",
       po::value(&image_file)->required(),
       "Path to the image file");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  CalibrationData calib(RawCalibrationData::read(calib_file));

  auto src_img = cv::imread(image_file, cv::IMREAD_GRAYSCALE);
  
  cv::Mat left(frame_height, frame_width, CV_8UC1); 
  cv::Mat right(frame_height, frame_width, CV_8UC1);

  cv::remap(
        src_img.colRange(0, frame_width), 
        left, 
        calib.undistortMaps[0].x, 
        calib.undistortMaps[0].y, 
        cv::INTER_LINEAR);

  cv::remap(
        src_img.colRange(frame_width, frame_width*2), 
        right, 
        calib.undistortMaps[1].x, 
        calib.undistortMaps[1].y, 
        cv::INTER_LINEAR);

  std::cout << left.size() << std::endl;

//  cv::imshow("image", src_img); 
//  cv::imshow("left", left);
//  cv::imshow("right", right);
//  cv::waitKey(-1);


  std::vector<cv::Point2f> left_corners, right_corners;

  if (!cv::findChessboardCorners(left, cv::Size(7, 5), left_corners)) {
    std::cout << "Left corners not found" << std::endl;
    return 0;
  }
  if (!cv::findChessboardCorners(right, cv::Size(7, 5), right_corners)) {
    std::cout << "Right corners not found" << std::endl;
    return 0;
  }

  cv::Mat debug_img(frame_height, frame_width*2, CV_8UC3);
  auto dbg_left = debug_img.colRange(0, frame_width);
  auto dbg_right = debug_img.colRange(frame_width, frame_width*2);
  cv::cvtColor(left, dbg_left, CV_GRAY2RGB);
  cv::cvtColor(right, dbg_right, CV_GRAY2RGB);

  cv::drawChessboardCorners(dbg_left, cv::Size(7, 5), left_corners, true);
  cv::drawChessboardCorners(dbg_right, cv::Size(7, 5), right_corners, true);

  std::vector<cv::Point3d> points;

  assert(left_corners.size() == right_corners.size());

  for (int i=0; i < left_corners.size(); ++i) {
    std::cout << left_corners[i].y - right_corners[i].y << std::endl;
    points.push_back(
        cv::Point3d(
            left_corners[i].x, 
            (left_corners[i].y + right_corners[i].y)/2.0,
            left_corners[i].x - right_corners[i].x));
  }

  cv::perspectiveTransform(points, points, calib.Q);

  auto x = cv::Mat_<double>(points.size(), 3);
  auto y = cv::Mat_<double>(points.size(), 1);

  for (int i=0; i < points.size(); ++i) {
    x(i, 0) = points[i].x;
    x(i, 1) = points[i].y;
    x(i, 2) = 1;
    y(i, 0) = points[i].z;
  }

  auto plane = (x.t()*x).inv()*x.t()*y;
  std::cout << plane << std::endl;

  cv::namedWindow("image");
  cv::imshow("image", debug_img); 
  cv::waitKey(-1);
}
