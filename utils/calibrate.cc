#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>

#include "calibration_data.hpp"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace boost::filesystem;
 
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

const int calib_flags = cv::CALIB_RATIONAL_MODEL;

typedef std::vector<std::vector<cv::Point2f>> Corners;
typedef std::vector<cv::Mat> Images;

bool detectCorners(
    const std::string& snapshots_dir,
    cv::Size chessboard_size,
    double chessboard_side_mm,
    cv::Size& img_size,
    Corners* left_corners,
    Corners* right_corners,
    std::vector<cv::Mat>* left_images,
    std::vector<cv::Mat>* right_images) {
  fs::path path(snapshots_dir);

  fs::directory_iterator end_iter;

  for (
      fs::directory_iterator it(path);
      it != end_iter;
      ++it) {
    auto f = *it;

    if (f.path().extension() != ".bmp") {
      continue;
    }

    auto full_img = cv::imread(f.path().c_str(), cv::IMREAD_GRAYSCALE);
    int w = full_img.cols/2;

    for (int camera_index=0; camera_index < 2; ++camera_index) {
      auto all_corners = camera_index == 0 ? left_corners : right_corners;
      if (all_corners != nullptr) {
        auto all_images = camera_index == 0 ? left_images : right_images;

        auto img = full_img.colRange(w*camera_index, w*(camera_index+1));

        if (all_images != nullptr) {
          all_images->push_back(img);
        }

        cv::Size cur_size(img.cols, img.rows);
        assert(img_size == cv::Size() || cur_size == img_size);
        img_size = cur_size;

        std::vector<cv::Point2f> corners;
        if (!cv::findChessboardCorners(
              img, chessboard_size, corners)) {
          std::cout << "Failed to find chessboard corners in " 
            << f.path() << std::endl;
          return false;
        }

        cv::cornerSubPix(
            img, 
            corners, 
            cv::Size(11, 11), 
            cv::Size(-1, -1), 
            cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 ));

        all_corners->push_back(corners);
      }
    }
  }

  return true;
}
    

bool calibrateCamera(
    const std::string& snapshots_dir,
    cv::Size chessboard_size,
    double chessboard_side_mm,
    int camera_index,
    cv::Size& img_size,
    cv::Mat& cameraMatrix,
    cv::Mat& distCoeffs,
    int debug = 0) {
  Corners all_corners;
  Images all_images;

  std::vector<cv::Mat> rvecs, tvecs;

  if (!detectCorners(
        snapshots_dir, 
        chessboard_size, 
        chessboard_side_mm,
        img_size,
        camera_index == 0 ? &all_corners : nullptr,
        camera_index == 1 ? &all_corners : nullptr,
        camera_index == 0 ? &all_images : nullptr,
        camera_index == 1 ? &all_images : nullptr)) {
    return false;
  }


  double residual = cv::calibrateCamera(
      modelCorners(all_corners.size(), chessboard_size, chessboard_side_mm),
      all_corners,
      img_size,
      cameraMatrix, 
      distCoeffs,
      rvecs,
      tvecs,
      calib_flags,
      cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1E-6));

  std::cout << "Reprojection error for camera " << camera_index << ": "
    << residual << std::endl;

  if (debug & DEBUG_ORIGINAL) {
    for (int i=0; i < (int)all_images.size(); ++i) {
      cv::Mat dbg_img;
      cv::cvtColor(all_images[i], dbg_img, CV_GRAY2RGB);
      cv::drawChessboardCorners(
          dbg_img, chessboard_size, all_corners[i], true);

      cv::imshow("debug", dbg_img);
      cv::waitKey(-1);
    }
  }

  if (debug & DEBUG_UNDISTORTED) {
    cv::Mat undistorted_img(img_size.height, img_size.width, CV_8UC1);
    cv::Mat newCameraMat = 
      cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, img_size, 1);

    for (auto img : all_images) {
      cv::undistort(
          img, undistorted_img, cameraMatrix, distCoeffs, newCameraMat);

      cv::imshow("debug", undistorted_img);
      cv::waitKey(-1);
    }
  }

  return true;
}

bool calibrateStereo(
    const std::string& snapshots_dir,
    cv::Size chessboard_size,
    int chessboard_side_mm,
    cv::Size& img_size, 
    cv::Mat& leftM, cv::Mat& leftD,
    cv::Mat& rightM, cv::Mat& rightD,
    cv::Mat& R, cv::Mat& T,
    Corners& left_corners, Corners& right_corners) {

  Images left_images, right_images;

  if (!detectCorners(
        snapshots_dir,
        chessboard_size,
        chessboard_side_mm,
        img_size,
        &left_corners,
        &right_corners,
        &left_images,
        &right_images)) {
    return false;
  }

  assert(left_corners.size() == right_corners.size());

  for (int i = 0; i < (int)left_corners.size(); ++i) {
    cv::Mat leftR, leftT, rightR, rightT, tmp;

    auto modelPoints = modelCorners(1, chessboard_size, chessboard_side_mm).front();

    int resL = cv::solvePnP(
        modelPoints,
        left_corners[i],
        leftM, leftD,
        tmp, leftT);
    cv::Rodrigues(tmp, leftR);

    int resR = cv::solvePnP(
        modelPoints,
        right_corners[i],
        rightM, rightD,
        tmp, rightT);
    cv::Rodrigues(tmp, rightR);

    cv::Mat R = rightR*leftR.inv();
    cv::Mat t = rightT - R*leftT;

    cv::Rodrigues(R, tmp);

    double residual = 0;

    std::vector<cv::Point2f> left_img_points;

    cv::projectPoints(
        modelPoints,
        leftR, leftT,
        leftM, leftD,
        left_img_points);

    for (int j=0; j < (int)left_img_points.size(); ++j) {
      residual += cv::norm(left_img_points[j] - left_corners[i][j]);
    }

    std::vector<cv::Point2f> right_img_points;

    cv::projectPoints(
        modelPoints,
        rightR, rightT,
        rightM, rightD,
        right_img_points);

    for (int j=0; j < (int)left_img_points.size(); ++j) {
      residual += cv::norm(right_img_points[j] - right_corners[i][j]);
    }

    residual /= right_img_points.size() + left_img_points.size();
    residual = sqrt(residual);

    cv::Mat dbg_img;
    cv::cvtColor(left_images[i], dbg_img, CV_GRAY2RGB);
    cv::drawChessboardCorners(
        dbg_img, chessboard_size, left_img_points, true);

    cv::imshow("debug", dbg_img);
    cv::waitKey(-1);


    std::cout << i << ": " 
        << tmp << ", " << cv::norm(tmp) << ", " << t
//      << resL << ", " << leftR << ", " << leftT
//      << resR << ", " << rightR << ", " << rightT
        << residual 
      << std::endl;
  }

  cv::Mat E, F;
  double residual = cv::stereoCalibrate(
      modelCorners(left_corners.size(), chessboard_size, chessboard_side_mm),
      left_corners,
      right_corners,
      leftM, leftD,
      rightM, rightD,
      img_size,
      R, T, E, F,
      cv::CALIB_FIX_INTRINSIC | calib_flags);
  
  cv::Mat tmp;
  cv::Rodrigues(R, tmp);

  std::cout << tmp << std::endl;
  std::cout << T << std::endl;
  std::cout << "Stereo calibration residual: " << residual << std::endl;



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

  RawCalibrationData raw_calib;

  Corners left_corners, right_corners;

  if (!calibrateCamera(
        snapshots_dir + "/left", 
        chessboard_size, 
        chessboard_side_mm,
        0,
        raw_calib.size,
        raw_calib.Ml,
        raw_calib.dl/*,
        DEBUG_UNDISTORTED*/)) {
    std::cout << "Failed to calibrate left camera" << std::endl;
    return 1;
  }

  if (!calibrateCamera(
        snapshots_dir + "/right", 
        chessboard_size, 
        chessboard_side_mm,
        1,
        raw_calib.size,
        raw_calib.Mr,
        raw_calib.dr/*,
        DEBUG_UNDISTORTED*/)) {
    std::cout << "Failed to calibrate right camera" << std::endl;
    return 1;
  }

  if (!calibrateStereo(
        snapshots_dir + "/stereo",
        chessboard_size,
        chessboard_side_mm,
        raw_calib.size,
        raw_calib.Ml, raw_calib.dl,
        raw_calib.Mr, raw_calib.dr,
        raw_calib.R, raw_calib.T,
        left_corners, right_corners)) {
    std::cout << "Failed to calibrate stereo pair" << std::endl;
    return 1;
  }

  raw_calib.write(calib_file);

  CalibrationData calib(raw_calib);

  for (int i=0; i < (int)left_corners.size(); ++i) {
    std::vector<cv::Point2f> left, right;

    cv::undistortPoints(
        left_corners[i], left, 
        calib.raw.Ml, calib.raw.dl, calib.Rl, calib.Pl);
    
    cv::undistortPoints(
        right_corners[i], right, 
        calib.raw.Mr, calib.raw.dr, calib.Rr, calib.Pr);

//    for (int p=0; p < left.size(); ++p) {
//      std::cout << left[p].y - right[p].y << std::endl;
//    }
  }

  return 0;
}
