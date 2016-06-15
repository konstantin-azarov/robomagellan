#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <assert.h>

#include "average.hpp"
#include "backward.hpp"
#include "calibration_data.hpp"
#include "clique.hpp"
#include "cross_frame_processor.hpp"
#include "fps_meter.hpp"
#include "frame_processor.hpp"
#include "math3d.hpp"
#include "raw_video_reader.h"
#include "reprojection_estimator.hpp"
#include "rigid_estimator.hpp"
#include "timer.hpp"

using namespace std;

namespace po = boost::program_options;

const int frame_width = 640;
const int frame_height = 480;

void onMouse(int event, int x, int y, int flags, void* ptr) {
  const FrameProcessor* p = static_cast<FrameProcessor*>(ptr);

  if (event == cv::EVENT_LBUTTONDOWN) {
    //p->printKeypointInfo(x, y);
  }
}

backward::SignalHandling sh;

int main(int argc, char** argv) {
  string video_file, calib_file;
  int fps, start;

  po::options_description desc("Command line options");
  desc.add_options()
      ("video-file",
       po::value<string>(&video_file)->required(),
       "path to the video file");

  desc.add_options()
      ("fps",
       po::value<int>(&fps)->required(),
       "video frame rate");

  desc.add_options()
      ("start",
       po::value<int>(&start)->default_value(0),
       "start at this second");

  desc.add_options()
      ("calib-file",
       po::value<string>(&calib_file)->default_value("data/calib.yml"),
       "path to the calibration file");
 
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  
  CalibrationData calib = CalibrationData::read(calib_file, frame_width, frame_height);

  FrameProcessor frame_processors[] = {
    FrameProcessor(calib),
    FrameProcessor(calib)
  };
  CrossFrameProcessor cross_processor(&calib.intrinsics);

  RawVideoReader rdr(video_file, frame_width*2, frame_height);
  uint8_t frame_data[frame_width*2*frame_height];
  cv::Mat frame_mat(frame_height, frame_width*2, CV_8UC1, frame_data);
  cv::Mat mono_frames[] = {
    frame_mat(cv::Range::all(), cv::Range(0, frame_width)),
    frame_mat(cv::Range::all(), cv::Range(frame_width, frame_width*2))
  };
  cv::Mat debug_mat(frame_height, frame_width*2, CV_8UC2);

  rdr.skip(fps*start);

  cv::namedWindow("video");
//  cv::setMouseCallback("video", onMouse, &processor);

  Timer timer;
  Average readTime(fps), 
          processTime(fps),
          renderTime(fps), 
          presentTime(fps);
  FpsMeter fpsMeter(fps);

  double frameTime = nanoTime();
  double frameDt = 1.0/fps;

  int frame_index = 0;

  cv::Mat camR = cv::Mat::eye(3, 3, CV_64F);
  cv::Point3d camT = cv::Point3d(0, 0, 0);

  bool done = false;
  while (!done) {
    timer.mark();

    if (!rdr.nextFrame(frame_data)) {
      break;
    }

    readTime.sample(timer.mark());

    FrameProcessor& processor = frame_processors[frame_index & 1];

    processor.process(mono_frames, debug_mat);

    processTime.sample(timer.mark());

    if (frame_index != 0) {
      if (cross_processor.process(
             frame_processors[!(frame_index & 1)],
             processor)) {
        camT = camR*cross_processor.t() + camT;
        camR = camR*cross_processor.rot();

        std::cout << "R = " << camR << endl;
        std::cout << "T = " << camT << endl;
      }

    }

    // Render
    char buf[1024];
    snprintf(
        buf, 
        sizeof(buf),
        "read_time = %.3f, render_time = %.3f, wait_time = %.3f process_time = %.3f fps = %.2f", 
        readTime.value(), 
        renderTime.value(),
        presentTime.value(),
        processTime.value(),
        fpsMeter.currentFps());

    cv::putText(
        debug_mat, 
        buf, 
        cv::Point(0, frame_height-10), 
        cv::FONT_HERSHEY_PLAIN, 
        1.0, 
        cv::Scalar(0, 255, 0));

    cv::imshow("video", debug_mat);

    renderTime.sample(timer.mark());
    fpsMeter.mark();

    // Wait for next frame
    frameTime += frameDt;
    //long wait = max(1, static_cast<int>((frameTime - nanoTime())*1000));

    int key = cv::waitKey(1);
    if (key != -1) {
      key &= 0xFF;
    }
    switch(key) {
      case 27:
        done = true;
        break;
    }

    presentTime.sample(timer.mark());

    frame_index++;
  }

  return 0;
}
