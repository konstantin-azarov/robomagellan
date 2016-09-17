#include <chrono>
#include <iostream>

#include <assert.h>
#include <gflags/gflags.h>
#include <opencv2/core.hpp>

#include "bag_video_reader.hpp"
#include "calibration_data.hpp"
#include "viso_stereo.h"

DEFINE_string(calib_file, "", "File with calibration");
DEFINE_string(video_file, "", "Video file to process");
DEFINE_int32(starting_frame, 0, "Start with this frame");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  CalibrationData calib(RawCalibrationData::read(FLAGS_calib_file)); 
  BagVideoReader reader(FLAGS_video_file, "/image_raw");


  VisualOdometryStereo::parameters param;

  param.calib.f = calib.intrinsics.f;
  param.calib.cu = calib.intrinsics.cxl;
  assert(calib.intrinsics.cxl == calib.intrinsics.cxr);
  param.calib.cv = calib.intrinsics.cy;
  param.base = -calib.intrinsics.dr / calib.intrinsics.f / 1000;

  std::cout << "Base = " << param.calib.cu << " " << param.calib.cv << std::endl;

  VisualOdometryStereo viso(param);

  Matrix pose = Matrix::eye(4);
  
  int32_t frame_width = calib.raw.size.width;
  int32_t frame_height = calib.raw.size.height;

  cv::Mat frame_mat(frame_height, frame_width*2, CV_8UC1);
  cv::Mat mono_frames[] = {
    frame_mat(cv::Range::all(), cv::Range(0, frame_width)),
    frame_mat(cv::Range::all(), cv::Range(frame_width, frame_width*2))
  };

  reader.skip(FLAGS_starting_frame);

  bool done = false;
  int i = FLAGS_starting_frame;

  while (!done) {
    if (!reader.nextFrame(frame_mat)) {
      break;
    }

    int32_t dims[] = { frame_width, frame_height, (int32_t)frame_mat.step };

    std::cout << "Processing: Frame: " << i;
  
    auto t0 = std::chrono::high_resolution_clock::now();
    if (viso.process(mono_frames[0].data, mono_frames[1].data, dims)) {
      auto t1 = std::chrono::high_resolution_clock::now();
        // on success, update current pose
        pose = pose * Matrix::inv(viso.getMotion());
      
        // output some statistics
        double num_matches = viso.getNumberOfMatches();
        double num_inliers = viso.getNumberOfInliers();
        std::cout << ", dt = " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << ", Matches: " << num_matches;
        std::cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << std::endl;
        std::cout << pose << std::endl << std::endl;
    } else {
      std::cout << " ... failed!" << std::endl;
    }

    ++i;
  }

  return 0;
}
