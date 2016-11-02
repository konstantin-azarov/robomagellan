#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
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

struct CalibrationException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

 
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

const int calib_flags = 0; //cv::CALIB_RATIONAL_MODEL;

struct Image {
  Image(const cv::Mat& data1, const std::string& filename1) : 
      data(data1), 
      filename(filename1) {
  };

  cv::Mat data;
  std::string filename;
};

typedef std::vector<cv::Point2f> Corners;
typedef std::vector<Corners> AllCorners;
typedef std::vector<Image> Images;

Images readImages(const std::string& dir) {
  fs::path path(dir);

  fs::directory_iterator end_iter;

  Images res;
  for (
      fs::directory_iterator it(path);
      it != end_iter;
      ++it) {
    auto f = *it;

    if (f.path().extension() != ".bmp" && f.path().extension() != ".jpg") {
      continue;
    }

    res.push_back(
        Image(
          cv::imread(f.path().c_str(), cv::IMREAD_GRAYSCALE),
          f.path().c_str()));
  }

  return res;
}

std::pair<Images, Images> splitImages(Images images) {
  Images left, right;

  for (auto img : images) {
    int w = img.data.cols / 2;
    left.push_back(Image(img.data.colRange(0, w), img.filename + "#left"));
    right.push_back(Image(img.data.colRange(w, 2*w), img.filename + "#right"));
  }

  return make_pair(left, right);
}

Corners detectCorners(const Image& image, cv::Size chessboard_size) {
  std::vector<cv::Point2f> corners;
  if (!cv::findChessboardCorners(
        image.data, chessboard_size, corners)) {
    throw CalibrationException(
        "Failed to find chessboard corners in " + image.filename);
  }

  cv::cornerSubPix(
      image.data, 
      corners, 
      cv::Size(11, 11), 
      cv::Size(-1, -1), 
      cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 ));

  return corners;
}

AllCorners detectCorners(
    const Images& images, 
    cv::Size chessboard_size,
    cv::Size& img_size) {
  AllCorners corners;

  std::cout << "Detecting corners: " << std::endl;
  for (const auto& img : images) {
  std::cout << "  " << img.filename << std::endl;
    corners.push_back(detectCorners(img, chessboard_size));

    if (img_size.width != 0 && img.data.size() != img_size) {
      throw CalibrationException("Image size mismatch");
    }
    img_size = img.data.size();
  }
  std::cout << std::endl;

  return corners;
}

double calibrateCamera(
    cv::Size img_size,
    cv::Size chessboard_size,
    double chessboard_side_mm,
    const AllCorners& corners,
    CameraCalibrationData& calibration) {

  std::vector<cv::Mat> rvecs, tvecs;

  double residual = cv::calibrateCamera(
      modelCorners(corners.size(), chessboard_size, chessboard_side_mm),
      corners,
      img_size,
      calibration.m, 
      calibration.d,
      rvecs,
      tvecs,
      calib_flags,
      cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1E-6));

  return residual;
}

double calibrateStereo(
    cv::Size img_size, 
    cv::Size chessboard_size,
    int chessboard_side_mm,
    const AllCorners& left_corners,
    const AllCorners& right_corners,
    CameraCalibrationData left_camera,
    CameraCalibrationData right_camera,
    cv::Mat& R, cv::Mat& T) {

  assert(left_corners.size() == right_corners.size());

  for (int i = 0; i < (int)left_corners.size(); ++i) {
    cv::Mat leftR, leftT, rightR, rightT, tmp;

    auto modelPoints = modelCorners(1, chessboard_size, chessboard_side_mm).front();

    cv::solvePnP(
        modelPoints,
        left_corners[i],
        left_camera.m, left_camera.d,
        tmp, leftT);
    cv::Rodrigues(tmp, leftR);

    cv::solvePnP(
        modelPoints,
        right_corners[i],
        right_camera.m, right_camera.d,
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
        left_camera.m, left_camera.d,
        left_img_points);

    for (int j=0; j < (int)left_img_points.size(); ++j) {
      residual += cv::norm(left_img_points[j] - left_corners[i][j]);
    }

    std::vector<cv::Point2f> right_img_points;

    cv::projectPoints(
        modelPoints,
        rightR, rightT,
        right_camera.m, right_camera.d,
        right_img_points);

    for (int j=0; j < (int)left_img_points.size(); ++j) {
      residual += cv::norm(right_img_points[j] - right_corners[i][j]);
    }

    residual /= right_img_points.size() + left_img_points.size();
    residual = sqrt(residual);

    /* cv::Mat dbg_img(img_size.height, 2*img_size.width, CV_8UC3); */
    /* cv::cvtColor(left_images[i], dbg_img.colRange(0, img_size.width), CV_GRAY2RGB); */
    /* cv::drawChessboardCorners( */
    /*     dbg_img.colRange(0, img_size.width), chessboard_size, left_corners[i], true); */
    
    /* cv::cvtColor(right_images[i], dbg_img.colRange(img_size.width, img_size.width*2), CV_GRAY2RGB); */
    /* cv::drawChessboardCorners( */
    /*     dbg_img.colRange(img_size.width, img_size.width*2), chessboard_size, right_img_points, true); */

//    cv::imshow("debug", dbg_img);
//    cv::waitKey(-1);


//    std::cout << i << ": " 
//        << tmp << ", " << cv::norm(tmp) << ", " << t
//      << resL << ", " << leftR << ", " << leftT
//      << resR << ", " << rightR << ", " << rightT
//        << residual 
//      << std::endl;
  }

  cv::Mat E, F;
  double residual = cv::stereoCalibrate(
      modelCorners(left_corners.size(), chessboard_size, chessboard_side_mm),
      left_corners,
      right_corners,
      left_camera.m, left_camera.d,
      right_camera.m, right_camera.d,
      img_size,
      R, T, E, F,
      cv::CALIB_FIX_INTRINSIC | calib_flags);
  
  return residual;
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

//  std::cout << lines << std::endl;

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

int main(int argc, char** argv) {
  std::string snapshots_dir, calib_file, mono_calib_file;
  cv::Size chessboard_size;
  double chessboard_side_mm;

  po::options_description desc("Command line options");
  desc.add_options()
      ("mono-calib-file",
       po::value(&mono_calib_file)->default_value("data/calib_phone.yml"),
       "calibrate one camera only")
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

  if (!mono_calib_file.empty()) {
    StereoCalibrationData stereo_calib(
        RawStereoCalibrationData::read(calib_file));
    RawMonoCalibrationData raw_calib;

    auto images = readImages(snapshots_dir);
    auto corners = detectCorners(images, chessboard_size, raw_calib.size);

    double residual = calibrateCamera(
        raw_calib.size,
        chessboard_size,
        chessboard_side_mm,
        corners,
        raw_calib.camera);

    std::cout << "Residual: " << residual << std::endl;

    raw_calib.write(calib_file);

    MonoCalibrationData calib(
        raw_calib, 
        stereo_calib.Pl.colRange(0, 3),
        stereo_calib.raw.size);

    cv::namedWindow("debug");
    for (const auto& img : images) {
      cv::Mat undistorted;

      cv::remap(
          img.data, 
          undistorted, 
          calib.undistort_maps.x, 
          calib.undistort_maps.y, 
          cv::INTER_LINEAR);

      cv::imshow("debug", undistorted);
      cv::waitKey(0);
    }

  } else {
    RawStereoCalibrationData raw_calib;

    auto left_images = 
      splitImages(readImages(snapshots_dir + "/left")).first;
    auto left_corners = detectCorners(
        left_images, chessboard_size, raw_calib.size);

    auto right_images = 
      splitImages(readImages(snapshots_dir + "/right")).second;
    auto right_corners = detectCorners(
        right_images, chessboard_size, raw_calib.size);

    auto stereo_images =
      splitImages(readImages(snapshots_dir + "/stereo"));
    auto stereo_corners = make_pair(
        detectCorners(stereo_images.first, chessboard_size, raw_calib.size),
        detectCorners(stereo_images.second, chessboard_size, raw_calib.size));

    double left_residual = calibrateCamera(
        raw_calib.size,
        chessboard_size, 
        chessboard_side_mm,
        left_corners,
        raw_calib.left_camera);

    double right_residual = calibrateCamera(
        raw_calib.size,
        chessboard_size, 
        chessboard_side_mm,
        right_corners,
        raw_calib.right_camera);
 
    double stereo_residual = calibrateStereo(
          raw_calib.size,
          chessboard_size,
          chessboard_side_mm,
          stereo_corners.first,
          stereo_corners.second,
          raw_calib.left_camera,
          raw_calib.right_camera,
          raw_calib.R, raw_calib.T);

    std::cout 
      << "Stereo calibration residuals: " << std::endl
      << "  left = " << left_residual << std::endl
      << "  right = " << right_residual << std::endl
      << "  stereo = " << stereo_residual << std::endl;

    raw_calib.write(calib_file);

    StereoCalibrationData calib(raw_calib);

    for (int i=0; i < (int)left_corners.size(); ++i) {
      std::vector<cv::Point2f> left, right;

      cv::undistortPoints(
          stereo_corners.first[i], left, 
          calib.raw.left_camera.m, calib.raw.right_camera.d, calib.Rl, calib.Pl);
      
      cv::undistortPoints(
          stereo_corners.second[i], right, 
          calib.raw.right_camera.m, calib.raw.right_camera.d, calib.Rr, calib.Pr);
    
      int w = raw_calib.size.width;
      cv::Mat dbg_img(raw_calib.size.height, 2*w, CV_8UC3);
      cv::Mat dbg_left(dbg_img.colRange(0, w));
      cv::Mat dbg_right(dbg_img.colRange(w, w*2));
      cv::Mat tmp;

      cv::remap(
          stereo_images.first[i].data, 
          tmp, 
          calib.undistort_maps[0].x, calib.undistort_maps[0].y, 
          cv::INTER_LINEAR);
      cv::cvtColor(tmp, dbg_left, CV_GRAY2RGB);

      cv::remap(
          stereo_images.second[i].data, 
          tmp, 
          calib.undistort_maps[1].x, calib.undistort_maps[1].y, 
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

      auto points_left_n = normalizePoints(points_left, raw_calib.left_camera.m);
      auto points_right_n = normalizePoints(points_right, raw_calib.right_camera.m);

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
  }

  return 0;
}
