#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <string>
#include <iostream>
#include <iomanip>

#include "camera.hpp"
#include "utils.hpp"

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

using namespace std;
using boost::format;
using boost::str;

namespace po = boost::program_options;

const int FRAME_WIDTH = 1280;
const int FRAME_HEIGHT = 720;
const int FRAME_SIZE = FRAME_WIDTH*FRAME_HEIGHT;
const int FPS = 30;

backward::SignalHandling sh;

void fail(const char* msg) {
  cerr << msg << endl;
  exit(1);
}

FILE* open_video_sink(string video_format, string output_file) {
  FILE* video_sink = nullptr;
  if (video_format == "x264") {
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

    video_sink = popen(cmd.c_str(), "w");
    if (video_sink == 0) {
      fail("Failed to open ffmpeg");
    }
  } else if (video_format == "raw") {
    video_sink = fopen(output_file.c_str(), "wb");
    if (video_sink == 0) {
      fail("Failed to open output file");
    }
  } else {
    fail("Invalid video format");
  }

  return video_sink;
}

void close_video_sink(string video_format, FILE* video_sink) {
  if (video_sink != nullptr) {
    if (video_format == "x264"){
      pclose(video_sink);
    } else {
      fclose(video_sink);
    }
  }
}

int main(int argc, char **argv) {
  string snapshots_dir, output_file, video_format;
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
  desc.add_options()
      ("format",
       po::value<string>(&video_format)->default_value("raw"),
       "raw or x264");

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

  if (video_format != "raw" && video_format != "x264") {
    fail("--video_format should be either 'raw' or 'x264'");
  }

  Camera camera;
  if (!camera.init(FRAME_WIDTH*2, FRAME_HEIGHT, FPS)) {
    fail("Failed to initialize camera");
  }

  FILE* video_sink = nullptr;

#ifndef NO_PREVIEW
  cv::namedWindow("preview");
#else
  if (!output_file.empty()) {
    video_sink = open_video_sink(video_format, output_file);
  }
#endif
 
#define SHOW_PARAM(name) \
  auto cur##name = camera.get##name(); \
  decltype(cur##name) min##name, max##name; \
  camera.get##name##Limits(min##name, max##name); \
  std::cout << #name \
      << ": min = " << min##name \
      << " max = " << max##name \
      << " cur = " << cur##name \
      << std::endl;

  SHOW_PARAM(Gain);
  SHOW_PARAM(Exposure);

  cout << "Ready" << endl;

  uint8_t buffer[FRAME_SIZE*4];

  cv::Mat raw(FRAME_HEIGHT, FRAME_WIDTH*2, CV_8UC2, buffer);
  cv::Mat combined(FRAME_HEIGHT, FRAME_WIDTH*2, CV_8UC3);
  cv::Mat preview;

  int current_snapshot_index = 0;
  bool done = false;
  int frame_count = 0;
  double last_timestamp = nanoTime();
  double t0 = last_timestamp;
  int fps_frame_count = 0;
  while(!done) {
    camera.nextFrame(buffer);

    cv::cvtColor(raw, combined, CV_YUV2RGB_YVYU);

    frame_count++;

#ifndef NO_PREVIEW
    cv::resize(combined, preview, cv::Size(0, 0), 0.5, 0.5);
    cv::imshow("preview", preview);
#endif

    if (video_sink != nullptr) {
      fwrite(buffer, 1, FRAME_SIZE*2, video_sink);
    }

    fps_frame_count++;
    double t = nanoTime();
    if (t - last_timestamp >= 2) {
      cout << setprecision(3) << "t = " << t - t0
        << " FPS = " << (fps_frame_count / (t - last_timestamp)) 
        << endl;
      last_timestamp = t;
      fps_frame_count = 0;

      cout << "Exposure: " << camera.getExposure() << endl;
//      cout << "Gain: " << camera.getGain() << endl;
    }

    if (video_duration != -1 && frame_count >= video_duration*FPS) {
      done = true;
    }

#ifndef NO_PREVIEW
    int key = cv::waitKey(1);
    if (key != -1) {
      key &= 0xFF;   // In ubuntu it returns some sort of a long key.
    }
    switch (key) {
      case 27:
        done = true;
        break;
      case 32:
        printf("Snapshot %d!\n", ++current_snapshot_index);

        cv::imwrite(
            str(format("%s/snapshot_%d.bmp") % 
              snapshots_dir % 
              current_snapshot_index), 
            combined);
        break;
      case 'r':
        if (video_sink == nullptr) {
          cout << "Recording" << endl;
          video_sink = open_video_sink(video_format, output_file);
        } else {
          cout << "Stopped recording" << endl;
          close_video_sink(video_format, video_sink);
          video_sink = nullptr;
        }
        break;
      case '=':
      case '+':
        curExposure = min(maxExposure, curExposure + (key == '=' ? 50 : 1000));
        cout << "Exposure: cur = " << curExposure << endl;
        camera.setExposure(curExposure);
        break;
      case '-':
      case '_':
        curExposure = max(minExposure, curExposure - (key == '-' ? 50 : 1000));
        cout << "Exposure: cur = " << curExposure << endl;
        camera.setExposure(curExposure);
        break;
      case 226:
        break; // shift
      case -1:
        break;
      default:
        printf("Unknown code %d\n", key);
        break;
    }
#endif
  }

  cout << endl;

  close_video_sink(video_format, video_sink);
  camera.shutdown();

  return 0;
}
