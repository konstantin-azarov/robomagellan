#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>

#include "calibration_data.hpp"

namespace fs = boost::filesystem;
namespace po = boost::program_options;
 
std::vector<std::vector<cv::Point3f>> modelCorners(
    int cnt, cv::Size sz, double side) {
  std::vector<std::vector<cv::Point3f>> res(cnt);

  for (int i=0; i < cnt; ++i) {
    for (int y=0; y < sz.height; ++y) {
      for (int x=0; x < sz.width; ++x) {
        res[i].push_back(cv::Point3f(x*side, y*side, 0));
      }
    }
  }

  return res;
}

const int DEBUG_ORIGINAL = 1;
const int DEBUG_UNDISTORTED = 2;

bool calibrateCamera(
    const std::string& snapshots_dir,
    cv::Size chessboard_size,
    double chessboard_side_mm,
    int camera_index,
    cv::Mat& cameraMatrix,
    cv::Mat& distCoeffs,
    int debug = 0) {
  fs::path path(snapshots_dir);

  std::vector<std::vector<cv::Point2f>> all_corners;
  std::vector<cv::Mat> all_images;

  cv::Size img_size;

  for (auto f : fs::directory_iterator(fs::path(snapshots_dir))) {
    if (f.path().extension() != ".bmp") {
      continue;
    }


    auto full_img = cv::imread(f.path().c_str(), cv::IMREAD_GRAYSCALE);
    int w = full_img.cols/2;
    auto img = full_img.colRange(w*camera_index, w*(camera_index+1));

    all_images.push_back(img);

    cv::Size cur_size(img.cols, img.rows);
    assert(img_size == cv::Size() || cur_size == img_size);
    img_size = cur_size;

    std::vector<cv::Point2f> corners;
    if (!cv::findChessboardCorners(
          img, chessboard_size, corners)) {
      std::cout << "Failed to find chessboard corners" << std::endl;
    }

    cv::cornerSubPix(
        img, 
        corners, 
        cv::Size(11, 11), 
        cv::Size(-1, -1), 
        cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 ));

    all_corners.push_back(corners);

    if (debug & DEBUG_ORIGINAL) {
      cv::Mat dbg_img;
      cv::cvtColor(img, dbg_img, CV_GRAY2RGB);
      cv::drawChessboardCorners(dbg_img, chessboard_size, corners, true);

      cv::imshow("debug", dbg_img);
      cv::waitKey(-1);
    }
  }

  std::vector<cv::Mat> rvecs, tvecs;

  double residual = cv::calibrateCamera(
      modelCorners(all_corners.size(), chessboard_size, chessboard_side_mm),
      all_corners,
      img_size,
      cameraMatrix, 
      distCoeffs,
      rvecs,
      tvecs,
      cv::CALIB_RATIONAL_MODEL,
      cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1E-6));

  std::cout << "Reprojection error for camera " << camera_index << ": "
    << residual << std::endl;

  if (debug & DEBUG_UNDISTORTED) {
    cv::Mat undistorted_img(img_size.height, img_size.width, CV_8UC1);
    cv::Mat newCameraMat = 
      cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, img_size, 1);

    for (auto img : all_images) {
      cv::undistort(img, undistorted_img, cameraMatrix, distCoeffs, newCameraMat);

      cv::imshow("debug", undistorted_img);
      cv::waitKey(-1);
    }
  }

  return true;
}


int main(int argc, char** argv) {
  std::string snapshots_dir, calib_file;
  cv::Size chessboard_size;
  double chessboard_side_mm;

  po::options_description desc("Command line options");
  desc.add_options()
      ("calib-file",
       po::value(&calib_file)->default_value("data/calib.yml"),
       "path to the calibration file")
      ("snapshots-dir",
       po::value(&snapshots_dir)->default_value("snapshots/calibration"),
       "directory with calibration snapshots")
      ("chessboard-witdh",
       po::value(&chessboard_size.width)->default_value(7),
       "chessboard width")
      ("chessboard-height",
       po::value(&chessboard_size.height)->default_value(5),
       "chessboard height")
      ("chessboard-side-mm",
       po::value(&chessboard_side_mm)->required());

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  RawCalibrationData calib;

  if (!calibrateCamera(
        snapshots_dir + "/left", 
        chessboard_size, 
        chessboard_side_mm,
        0,
        calib.Ml,
        calib.dl)) {
    std::cout << "Failed to calibrate left camera" << std::endl;
    return 1;
  }

  if (!calibrateCamera(
        snapshots_dir + "/right", 
        chessboard_size, 
        chessboard_side_mm,
        1,
        calib.Mr,
        calib.dr,
        DEBUG_ORIGINAL | DEBUG_UNDISTORTED)) {
    std::cout << "Failed to calibrate right camera" << std::endl;
    return 1;
  }
}
