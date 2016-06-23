#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <assert.h>

#include "average.hpp"
#include "backward.hpp"
#include "calibration_data.hpp"
#include "clique.hpp"
#include "cross_frame_processor.hpp"
#include "debug_renderer.hpp"
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
  auto p = static_cast<DebugRenderer*>(ptr);

  if (event == cv::EVENT_LBUTTONDOWN) {
    p->selectKeypoint(x, y);
    cv::imshow("video", p->debugImage());
  }
}

backward::SignalHandling sh;

int main(int argc, char** argv) {
  string video_file, calib_file;
  int start;
  bool debug_single_frame;

  po::options_description desc("Command line options");
  desc.add_options()
      ("video-file",
       po::value<string>(&video_file)->required(),
       "path to the video file");

  desc.add_options()
      ("start-frame",
       po::value<int>(&start)->default_value(0),
       "start at this second");

  desc.add_options()
      ("calib-file",
       po::value<string>(&calib_file)->default_value("data/calib.yml"),
       "path to the calibration file");

  desc.add_options()
      ("debug-single-frames",
       po::value<bool>(&debug_single_frame)->default_value(false),
       "debug single frame (at second --start)");
 
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  
  CalibrationData calib = CalibrationData::read(calib_file, frame_width, frame_height);

  RawVideoReader rdr(video_file, frame_width*2, frame_height);
  cv::Mat frame_mat(frame_height, frame_width*2, CV_8UC1);
  cv::Mat mono_frames[] = {
    frame_mat(cv::Range::all(), cv::Range(0, frame_width)),
    frame_mat(cv::Range::all(), cv::Range(frame_width, frame_width*2))
  };

  DebugRenderer debug_renderer;

  rdr.skip(start);
    
  int global_frame_index = start;

  cv::namedWindow("video");
  if (debug_single_frame) {
    cv::setMouseCallback("video", onMouse, &debug_renderer);

    FrameProcessor processor(calib);
    if (!rdr.nextFrame(frame_mat.data)) {
      std::cout << "No frame to read" << std::endl;
      return 0;
    }


    bool done = false;
    bool show_features = false;
    bool show_matches = false;
    int threshold = 50;

    while (!done) {
      processor.process(mono_frames, threshold);

      debug_renderer.start(&processor, &processor, nullptr);
      if (show_features) {
        debug_renderer.renderFeatures();
      }
      if (show_matches) {
        debug_renderer.renderMatches();
      }
     
      char buf[1024];
      snprintf(
          buf, 
          sizeof(buf),
          "frame = %d; thr = %d; features: left = %ld, right = %ld; points = %ld", 
          global_frame_index, threshold,
          processor.keypoints(0).size(), 
          processor.keypoints(1).size(),
          processor.points().size());

      debug_renderer.renderText(std::string(buf));
      cv::imshow("video", debug_renderer.debugImage());

      int key = cv::waitKey(-1);
      if (key != -1) {
        key &= 0xFF;
      }
      switch(key) {
        case ']':
          rdr.nextFrame(frame_mat.data);
          global_frame_index++;
          break;
        case '[':
          rdr.skip(-2);
          rdr.nextFrame(frame_mat.data);
          global_frame_index--;
          break;
        case '+':
          threshold++;
          break;
        case '-':
          threshold--;
          break;
        case 'f':
          show_features = !show_features;
          break;
        case 'm':
          show_matches = !show_matches;
          break;
        case 27:
          done = true;
          break;
      }
    }
  } else {
    FrameProcessor frame_processors[] = {
      FrameProcessor(calib),
      FrameProcessor(calib)
    };
    CrossFrameProcessor cross_processor(&calib.intrinsics);

    Timer timer;
    Average readTime(60), 
            processTime(60),
            renderTime(60), 
            presentTime(60);
    FpsMeter fpsMeter(60);

    double frameTime = nanoTime();
    double frameDt = 1.0/60.0;

    int frame_index = 0;

    cv::Mat camR = cv::Mat::eye(3, 3, CV_64F);
    cv::Point3d camT = cv::Point3d(0, 0, 0);

    bool done = false;
    while (!done) {
      timer.mark();

      if (!rdr.nextFrame(frame_mat.data)) {
        break;
      }

      readTime.sample(timer.mark());

      FrameProcessor& processor = frame_processors[frame_index & 1];

      processor.process(mono_frames, 30);

      processTime.sample(timer.mark());

      auto p1 = &frame_processors[!(frame_index & 1)];

      if (frame_index > 0) {
        if (cross_processor.process(*p1, processor)) {
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
          "f = %d, read_time = %.3f, render_time = %.3f, wait_time = %.3f process_time = %.3f fps = %.2f", 
          global_frame_index,
          readTime.value(), 
          renderTime.value(),
          presentTime.value(),
          processTime.value(),
          fpsMeter.currentFps());

      debug_renderer.start(p1, &processor, &cross_processor);
      debug_renderer.renderText(std::string(buf));
//      debug_renderer.renderMatches();
    //  debug_renderer.renderPointFeatures(326);
      debug_renderer.renderCrossMatches();

      cv::imshow("video", debug_renderer.debugImage());

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
      global_frame_index++;
    }
  }

  return 0;
}
