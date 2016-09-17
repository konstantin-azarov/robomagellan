#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros/console.h>
#include <zed/Camera.hpp>

namespace zed = sl::zed;

const int c_imgWidth = 1280;
const int c_imgHeight = 720;


int main(int argc, char** argv) {
  int rate = 15;

  zed::Camera camera(zed::ZEDResolution_mode::HD720, rate);

  zed::InitParams params;
  params.verbose = true;
  
  auto err = camera.init(params);

  if (err != zed::SUCCESS) {
    ROS_ERROR_STREAM("Failed to initialize camera: " << zed::errcode2str(err));
    return 1;
  }

  ros::Time::init();
  ros::Rate loop_rate(rate);

  cv::Mat combined_image(c_imgHeight, c_imgWidth*2, CV_8UC3);
  cv::Mat left_img = combined_image.colRange(0, c_imgWidth);
  cv::Mat right_img = combined_image.colRange(c_imgWidth, c_imgWidth*2);

  cv::Mat depth;
  
  cv::namedWindow("preview");

  auto p = camera.getParameters();
  auto l = p->LeftCam;
  auto r = p->RightCam;
  
  ROS_INFO_STREAM("Left: " << l.fx << " " << l.fy << " " << l.cx << " " << l.cy);
  ROS_INFO_STREAM("Left d: " << l.disto[0] << " " << l.disto[1] << " " << l.disto[2] << " " << l.disto[3] << " " << l.disto[4]);
  ROS_INFO_STREAM("Right: " << r.fx << " " << r.fy << " " << r.cx << " " << r.cy);
  ROS_INFO_STREAM("Right d: " << r.disto[0] << " " << r.disto[1] << " " << r.disto[2] << " " << r.disto[3] << " " << r.disto[4]);


  bool done = false;
  while (!done) {
    /* auto t0 = ros::Time::now(); */

    int err = camera.grab(zed::STANDARD, false, false, false);
    if (!err) {
      auto raw_left = camera.retrieveImage(zed::SIDE::LEFT_UNRECTIFIED);
      auto raw_right = camera.retrieveImage(zed::SIDE::RIGHT_UNRECTIFIED);

      cv::cvtColor(
          zed::slMat2cvMat(raw_left),
          left_img, 
          CV_RGBA2RGB);
      cv::cvtColor(
          zed::slMat2cvMat(raw_right),
          right_img,
          CV_RGBA2RGB);
   

      cv::imshow("preview", combined_image);

      int key = cv::waitKey(1);
      if (key != -1) {
        key &= 0xFF;
      }

      switch (key) {
        case 27:
          done = true;
          break;
        case 'e':
          ROS_INFO_STREAM("Exposure: " << camera.getCameraSettingsValue(zed::ZED_EXPOSURE));
      }
    }

    if (!loop_rate.sleep()) {
      ROS_INFO_STREAM("Missed frame");
    }
  }

  return 0;
}
