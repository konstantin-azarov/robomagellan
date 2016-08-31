#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "calibration_data.hpp"

#include "math3d.hpp"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace boost::filesystem;

using boost::format;
 
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

typedef std::vector<cv::Point2f> Corners;

class CalibrationException : public std::exception {
  public:
    CalibrationException(const std::string& error) : error_(error) {}

    virtual const char* what() const { 
      return error_.c_str();
    }
  
  private:
    std::string error_;
}

template <class T>
using Stereo = std::pair<T, T>;

template<class T, class U>
Stereo<U> map(std::function<U, T> f, const Stereo<T>& v) {
  return std::make_pair(f(v.first), f(v.second)); 
}

template<class T, class U>
std::vector<U> map(std::function<U, T> f, const std::vector<T>& v) {
  std::vector res;
  std::transform(v.begin(), v.end(), res.begin(), f);
  return res;
}

template<class T>
Stereo<std::vector<T>> unzip(const std::vector<Stereo<T>>& v) {
  Stereo<std::vector<T>> res;

  for (int i=0; i < v.size(); ++i) {
    res.first.push_back(v[i].first);
    res.second.push_back(v[i].second);
  }

  return res;
}

struct Image {
  Image(cv::Mat& data_, const std::string& path_) : data(data_), path(path_) {}

  cv::Mat data;
  std::string path;
};

struct CalibrationResult {
  cv::Size size;
  CameraCalibrationData calibration;
  std::vector<Image> images;
  std::vector<Corners> corners;
};

std::vector<Image> readImages(const std::string& path) {
  std::vector<Image> res;

  fs::path path(path);
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
    res.push_back(Image(full_img, f.path().c_str()));
  }

  return res;
}

Stereo<Image> splitImage(const Image img) {
  int w = img.data.cols/2;
  return std::make_pair(
      Image(img.data.colRange(0, w), img.path + ":left"), 
      Image(img.data.colRange(w, 2*w), img.path + ":right"));
}

Corners detectCorners(int chessboard_size, const cv::Mat& img) {
  Corners corners;

  if (!cv::findChessboardCorners(
        img, chessboard_size, corners)) {
    throw CalibrationException(
        format("Failed to find chessboard corners in %1%") % f.path());
  }

  cv::cornerSubPix(
      img, 
      corners, 
      cv::Size(11, 11), 
      cv::Size(-1, -1), 
      cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 ));

  return corners;
}
   
CameraCalibrationData computeCalibration(
    cv::Size chessboard_size,
    double chessboard_side_mm,
    cv::Size img_size,
    const std::string& camera_name,
    std::vector<Corners> corners) {

  CameraCalibrationData data;
  std::vector<cv::Mat> rvecs, tvecs;

  double residual = cv::calibrateCamera(
      modelCorners(corners.size(), chessboard_size, chessboard_side_mm),
      corners,
      img_size,
      data.M, 
      data.d,
      rvecs,
      tvecs,
      calib_flags,
      cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1E-6));

  std::cout << "Reprojection error for " << camera_name << " camera: "
    << residual << std::endl;

  return true;
}

CalibrationResult calibrateCamera(
    const cv::Size& chessboard_size,
    double chessboard_side_mm,
    const std::string& image_path,
    int camera_index) {
  CalibrationResult result;

  result.images = unzip(map(splitImage, readImages(image_path)))[camera_index];
  result.size = result.images[0].size;
  result.corners = map(std::bind(detectCorners, chessboard_size, _1), images);
  result.calibration = computeCalibration(
      chessboard_size, 
      chessboard_side_mm,
      img_size,
      camera_index == 0 ? "left" : "right",
      corners);
  
  return result;
}

StereoCalibrationData calibrateStereo(
    cv::Size chessboard_size,
    int chessboard_side_mm,
    const cv::Size& img_size, 
    const Stereo<CameraCalibrationData>& data,
    const Stereo<std::vector<Corners>>& corners) {

  for (int i = 0; i < (int)corners.first.size(); ++i) {
    cv::Mat leftR, leftT, rightR, rightT, tmp;

    auto modelPoints = 
      modelCorners(1, chessboard_size, chessboard_side_mm).front();

    cv::solvePnP(
        modelPoints,
        left_corners[i],
        leftM, leftD,
        tmp, leftT);
    cv::Rodrigues(tmp, leftR);

    cv::solvePnP(
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

    cv::Mat dbg_img(img_size.height, 2*img_size.width, CV_8UC3);
    cv::cvtColor(left_images[i], dbg_img.colRange(0, img_size.width), CV_GRAY2RGB);
    cv::drawChessboardCorners(
        dbg_img.colRange(0, img_size.width), chessboard_size, left_corners[i], true);
    
    cv::cvtColor(right_images[i], dbg_img.colRange(img_size.width, img_size.width*2), CV_GRAY2RGB);
    cv::drawChessboardCorners(
        dbg_img.colRange(img_size.width, img_size.width*2), chessboard_size, right_img_points, true);

//    cv::imshow("debug", dbg_img);
//    cv::waitKey(-1);


//    std::cout << i << ": " 
//        << tmp << ", " << cv::norm(tmp) << ", " << t
//      << resL << ", " << leftR << ", " << leftT
//      << resR << ", " << rightR << ", " << rightT
//        << residual 
//      << std::endl;
  }

  StereoCalibrationData data; 
  cv::Mat E, F;
  double residual = cv::stereoCalibrate(
      modelCorners(corners.first.size(), chessboard_size, chessboard_side_mm),
      corners.first.size(),
      corners.second.size(),
      data.first.M, data.first.d,
      data.second.M, data.second.d,
      img_size,
      data.R, data.T, E, F,
      cv::CALIB_FIX_INTRINSIC | calib_flags);
  
  cv::Mat tmp;
  cv::Rodrigues(data.R, tmp);

  std::cout << "R = " << tmp << std::endl;
  std::cout << "T = " << data.T << std::endl;
  std::cout << "Stereo calibration residual: " << residual << std::endl;

  return data;
}

cv::Vec2d findVanishingPoint(
    cv::InputArray corners_, 
    int chessboard_width,
    cv::Mat* dbg_img) {

  auto corners = corners_.getMat().reshape(1, corners_.cols());

  int n = corners.rows;
  assert((n % chessboard_width) == 0);

  cv::Mat lines(n/chessboard_width, 3, CV_64F);

  for (int s = 0, i = 0; s < n; s += chessboard_width, i++) {
    auto line = fitLine(corners.rowRange(s, s + chessboard_width));
    lines.row(i) = cv::Mat(line).t();

    if (dbg_img != nullptr) {
      double k = -line[0]/line[1];
      double b = -line[2]/line[1];

      cv::line(
          *dbg_img, 
          cv::Point(0, b), cv::Point(dbg_img->cols, dbg_img->cols*k + b),
          cv::Scalar(255, 0, 0));
    }
  }

  return intersectLines(lines);
}

std::pair<cv::Vec2d, cv::Vec2d> findVanishingPoints(
  std::vector<cv::Point2f> corners,
  int chessboard_width,
  cv::Mat* dbg_img) {
  
  auto horizontal = findVanishingPoint(corners, chessboard_width, dbg_img);
  int chessboard_height = corners.size() / chessboard_width;

  std::vector<cv::Point2f> transposed;
  for (int j = 0; j < chessboard_width; ++j) {
    for (int i = 0; i < chessboard_height; ++i) {
      transposed.push_back(corners[i*chessboard_width + j]);
    }
  }

  auto vertical = findVanishingPoint(transposed, chessboard_height, dbg_img);

  return std::make_pair(horizontal, vertical);
}

cv::Vec3d normalizePoint(const cv::Vec2d& p, const cv::Mat_<double>& m) {
  return cv::Vec3d((p[0] - m[0][2])/m[0][0], (p[1] - m[1][2])/m[1][1], 1.0);
}

std::pair<cv::Vec3d, cv::Vec3d> normalizePoints(
    const std::pair<cv::Vec2d, cv::Vec2d>& points,
    const cv::Mat_<double>& m) {
  return std::make_pair(
      normalizePoint(points.first, m),
      normalizePoint(points.second, m));
}

backward::SignalHandling sh;

void drawCross(cv::Mat& img, cv::Point pt, const cv::Scalar& color) {
  cv::line(img, cv::Point(pt.x - 5, pt.y), cv::Point(pt.x + 5, pt.y), color);
  cv::line(img, cv::Point(pt.x, pt.y - 5), cv::Point(pt.x, pt.y + 5), color);
}

//void testSingleCamera(int chessboard_size, double chessboard_side_mm) {
//  cv::Size size;
//  cv::Mat M, P;
//
//  if (!calibrateCamera(
//        "snapshots/test",
//        chessboard_size,
//        chessboard_side_mm,
//        -1,
//        size,
//        M,
//        d)) {
//    std::cerr << "Failed to calibrate single camera" << std::endl;
//    return;
//  }
//}

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

  // xcxc
//  testSingleCamera(chessboard_size, chessboard_side_mm);

  // end xcxc



  if (!calibrateCamera(
        snapshots_dir + "/left", 
        chessboard_size, 
        chessboard_side_mm,
        0,
        raw_calib.size,
        raw_calib.Ml,
        raw_calib.dl/*,
        DEBUG_ORIGINAL*/)) {
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

  AllCorners left_corners, right_corners;
  Images left_images, right_images;

  if (!calibrateStereo(
        snapshots_dir + "/stereo",
        chessboard_size,
        chessboard_side_mm,
        raw_calib.size,
        raw_calib.Ml, raw_calib.dl,
        raw_calib.Mr, raw_calib.dr,
        raw_calib.R, raw_calib.T,
        left_corners, right_corners,
        left_images, right_images)) {
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
  
    int w = raw_calib.size.width;
    cv::Mat dbg_img(raw_calib.size.height, 2*w, CV_8UC3);
    cv::Mat dbg_left(dbg_img.colRange(0, w));
    cv::Mat dbg_right(dbg_img.colRange(w, w*2));
    cv::Mat tmp;

    cv::remap(
        left_images[i], 
        tmp, 
        calib.undistortMaps[0].x, calib.undistortMaps[0].y, 
        cv::INTER_LINEAR);
    cv::cvtColor(tmp, dbg_left, CV_GRAY2RGB);

    cv::remap(
        right_images[i], 
        tmp, 
        calib.undistortMaps[1].x, calib.undistortMaps[1].y, 
        cv::INTER_LINEAR);
    cv::cvtColor(tmp, dbg_right, CV_GRAY2RGB);
    
    for (int j=0; j < (int)left.size(); ++j) {
      drawCross(dbg_left, left[j], cv::Scalar(0, 0, 255));
      drawCross(dbg_right, right[j], cv::Scalar(0, 0, 255));
    }

    auto points_left = 
      findVanishingPoints(left, chessboard_size.width, &dbg_left);
    auto points_right = 
      findVanishingPoints(right, chessboard_size.width, &dbg_right);

    auto points_left_n = normalizePoints(points_left, raw_calib.Ml);
    auto points_right_n = normalizePoints(points_right, raw_calib.Mr);

    std::cout << points_left_n.first.t() * points_left_n.second << std::endl;
    std::cout << points_right_n.first.t() * points_right_n.second << std::endl;

    std::cout 
      << "left = " << points_left.first << ", " << points_left.second 
      << "; right = " << points_right.first  << ", " << points_right.second
      << "; d = " << points_left.first - points_right.first << ", " 
      << points_left.second - points_right.second
      << std::endl;


    cv::imshow("debug", dbg_img);
    cv::waitKey(-1);
  }

  return 0;
}
