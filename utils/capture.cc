#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <string>
#include <iostream>
#include <ctime>

#include "camera.hpp"

using namespace std;
using boost::format;
using boost::str;

namespace po = boost::program_options;

const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
const int FRAME_SIZE = FRAME_WIDTH*FRAME_HEIGHT;
const int FPS = 30;

void uncombine(const uint8_t* data, uint8_t* left_data, uint8_t* right_data) {
  for (int i=0; i < FRAME_HEIGHT; ++i) {
    for (int j=0; j < FRAME_WIDTH; ++j) {
      *(left_data++) = *data;
      *(right_data++) = *(data + FRAME_WIDTH);
      data++;
    }
    data += FRAME_WIDTH;
  }
}

void fail(const char* msg) {
  cerr << msg << endl;
  exit(1);
}

int main(int argc, char **argv) {
  string snapshots_dir, output_file;
  int video_duration = -1;

  po::options_description desc("Command line options");
  desc.add_options()
      ("output-dir",
       po::value<string>(&snapshots_dir)->required(),
       "where to store snapshots");
  desc.add_options()
      ("output-file",
       po::value<string>(&output_file),
       "where to store outpit video file");
  desc.add_options()
      ("duration",
       po::value<int>(&video_duration),
       "how long to record the video for");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  #ifdef NO_PREVIEW
  if (output_file.empty()) {
    fail("Output file is required");
  }
  if (video_duration == -1) {
    fail("Video duration is required");
  }
  #endif

  Camera camera;
  camera.init(FRAME_WIDTH, FRAME_HEIGHT, FPS);

  FILE* ffmpeg = 0;

  if (!output_file.empty()) {
    std::string cmd = string("ffmpeg ") +
        "-f rawvideo " +
        "-pix_fmt gray " +
        "-s 1280x480 " +
        "-r 30 " +
        "-i - " +
        "-r 30 " +
        "-c:v libx264 " +
        "-preset ultrafast " +
        "-qp 0 " +
        "-an " +
        "-f avi " +
        "-y " +
        output_file;

    cout << "Running ffmpeg: " << cmd << endl;

    ffmpeg = popen(cmd.c_str(), "w");
    if (ffmpeg == 0) {
      fail("Failed to open ffmpeg");
    }
  }

#ifndef NO_PREVIEW
  cv::namedWindow("preview");
#endif

  uint8_t buffer[FRAME_SIZE*2];

  cv::Mat combined(FRAME_HEIGHT, FRAME_WIDTH*2, CV_8UC1, buffer);
  cv::Mat left(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);
  cv::Mat right(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);

  int current_snapshot_index = 0;
  bool done = false;
  int frame_count = 0;
  while(!done) {
    camera.nextFrame(buffer);
    frame_count++;

#ifndef NO_PREVIEW
    cv::imshow("preview", combined);
#endif

    if (ffmpeg != 0) {
      fwrite(buffer, 1, FRAME_SIZE*2, ffmpeg);
    }

    if (video_duration != -1 && frame_count >= video_duration*FPS) {
      done = true;
    }

#ifndef NO_PREVIEW
    int key = cv::waitKey(1);
    switch (key) {
      case 27:
        done = true;
        break;
      case 32:
        printf("Snapshot %d!\n", ++current_snapshot_index);

        uncombine(buffer, left.ptr(), right.ptr());
        cv::imwrite(str(format("%s/left_%d.png") % snapshots_dir % current_snapshot_index), left);
        cv::imwrite(str(format("%s/right_%d.png") % snapshots_dir % current_snapshot_index), right);

        break;
      case -1:
        break;
      default:
        printf("Unknown code %d\n", key);
        break;
    }
#endif
  }

  pclose(ffmpeg);

  camera.shutdown();

  return 0;
}
