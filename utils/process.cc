#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include "raw_video_reader.h"

using namespace std;

namespace po = boost::program_options;

const int frame_width = 640;
const int frame_height = 480;

int main(int argc, char** argv) {
  cv::FileStorage fs("test.yml", cv::FileStorage::WRITE);

  cv::Mat_<double> m1(3, 3), m2(3, 3);

  for (int i=0; i < 3; ++i) {
    m1(0, i) = i;
    m2(0, i) = -i;
  }

  fs << "m1" << m1 << "m2" << m2;
  fs.release();
  

  string video_file;
  int fps;

  po::options_description desc("Command line options");
  desc.add_options()
      ("video-file",
       po::value<string>(&video_file)->required(),
       "path to the video file");

  desc.add_options()
      ("fps",
       po::value<int>(&fps)->required(),
       "video frame rate");
  

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  RawVideoReader rdr(video_file, frame_width*2, frame_height);
  uint8_t frame_data[frame_width*2*frame_height];
  cv::Mat frame_mat(frame_height, frame_width*2, CV_8UC1, frame_data);

  cv::namedWindow("video");

  bool done = false;
  while (rdr.nextFrame(frame_data) && !done) {
    cv::imshow("video", frame_mat);

    int key = cv::waitKey(1000.0/fps);
    if (key != -1) {
      key &= 0xFF;
    }
    switch(key) {
      case 27:
        done = true;
        break;
    }
  }

  return 0;
}
