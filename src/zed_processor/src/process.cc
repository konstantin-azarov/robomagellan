#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <assert.h>

#include "average.hpp"
#define BACKWARD_HAS_DW 1
#include "backward.hpp"
#include "bag_video_reader.hpp"
#include "calibration_data.hpp"
#include "clique.hpp"
#include "cross_frame_processor.hpp"
#include "debug_renderer.hpp"
#include "fps_meter.hpp"
#include "frame_processor.hpp"
#include "math3d.hpp"
#include "reprojection_estimator.hpp"
#include "rigid_estimator.hpp"
#include "timer.hpp"

using namespace std;

namespace po = boost::program_options;

void onMouse(int event, int x, int y, int flags, void* ptr) {
  auto p = static_cast<DebugRenderer*>(ptr);

  if (event == cv::EVENT_LBUTTONDOWN) {
    p->selectKeypoint(x, y);
    cv::imshow("debug", p->debugImage());
  }
}

backward::SignalHandling sh;

int main(int argc, char** argv) {
  string video_file, calib_file;
  int start;
  bool debug;
  int frame_count;

  po::options_description desc("Command line options");
  desc.add_options()
      ("video-file",
       po::value<string>(&video_file)->required(),
       "path to the video file")
      ("start-frame",
       po::value<int>(&start)->default_value(0),
       "start at this second")
      ("total-frames",
       po::value<int>(&frame_count)->default_value(0),
       "number of frames to process")
      ("calib-file",
       po::value<string>(&calib_file)->default_value("data/calib.yml"),
       "path to the calibration file")
      ("debug",
       po::bool_switch(&debug)->default_value(false),
       "Start in debug mode.");
 
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Initialize GPU
  int gpu_count = cv::cuda::getCudaEnabledDeviceCount();
  std::cout << "Found " << gpu_count << " devices" << std::endl;
  for (int i=0; i < gpu_count; ++i) {
    cv::cuda::printCudaDeviceInfo(i);
  }

  StereoCalibrationData calib(RawStereoCalibrationData::read(calib_file));

  int frame_width = calib.raw.size.width;
  int frame_height = calib.raw.size.height;

  BagVideoReader rdr(video_file, "/image_raw");
  cv::Mat frame_mat(frame_height, frame_width*2, CV_8UC1);
  cv::Mat mono_frames[] = {
    frame_mat(cv::Range::all(), cv::Range(0, frame_width)),
    frame_mat(cv::Range::all(), cv::Range(frame_width, frame_width*2))
  };

  DebugRenderer debug_renderer(1600, 1200);

  rdr.skip(start);
    
  int global_frame_index = start-1;

//  cv::namedWindow("debug");
//  cv::setMouseCallback("debug", onMouse, &debug_renderer);

  FrameProcessor frame_processors[] = {
    FrameProcessor(calib),
    FrameProcessor(calib)
  };

  CrossFrameProcessorConfig cross_processor_config;
  CrossFrameProcessor cross_processor(calib, cross_processor_config);

  Timer timer;
  Average readTime(60), 
          processTime(60),
          renderTime(60), 
          presentTime(60);
  FpsMeter fpsMeter(60);

  double frameTime = nanoTime();
  double frameDt = 1.0/60.0;

  int frame_index = -1;
  int threshold = 25;

  bool show_features = false, 
       show_matches = false,
       show_cross_matches = false,
       show_filtered_matches = false,
       show_clique = false,
       show_clique_cross_features = false,
       show_inlier_features = false;
      

  cv::Mat camR = cv::Mat::eye(3, 3, CV_64F);
  cv::Point3d camT = cv::Point3d(0, 0, 0);

  bool done = false;
  while (!done && (frame_count == 0 || frame_index < frame_count)) {
    timer.mark();

    if (!rdr.nextFrame(frame_mat)) {
      break;
    }

    frame_index++;
    global_frame_index++;

    readTime.sample(timer.mark());
    
    std::cout << "Frame #" << global_frame_index << std::endl;

    FrameProcessor& processor = frame_processors[frame_index & 1];

    processor.process(mono_frames, threshold);

    processTime.sample(timer.mark());

    auto p1 = &frame_processors[!(frame_index & 1)];

    if (frame_index > 0) {
      bool ok = cross_processor.process(*p1, processor);

      if (ok) {
        const cv::Mat_<double> cov(cross_processor.t_cov());
//        ok &= cov[0][0] < 5 && cov[1][1] < 5 && cov[2][2] < 5;
        if (!(cov[0][0] < 5 && cov[1][1] < 5 && cov[2][2] < 5)) {
          std::cout << "FAIL";
          std::cout << "T_cov = " << cross_processor.t_cov() << std::endl; 
        } else {
          camT = cross_processor.rot()*camT + cross_processor.t();
          camR = cross_processor.rot()*camR;
        }

        /* std::cout << "R = " << camR << endl; */
        std::cout << "T = " << camT << endl; 
        /* std::cout << "T_cov = " << cross_processor.t_cov() << std::endl; */
      } else {
        std::cout << "FAIL" << std::endl;
      }
    }


    if (frame_index > 0 && debug) {
      while (!done) {
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
        if (debug) {
          debug_renderer.renderText(std::string(buf));
          if (show_features) {
            debug_renderer.renderFeatures();
          }

          if (show_matches) {
            debug_renderer.renderMatches();
          }

          if (show_cross_matches) {
            debug_renderer.renderAllCrossMatches();
          }

          if (show_filtered_matches) {
            debug_renderer.renderFilteredCrossMatches();
          }

          if (show_clique) {
            debug_renderer.renderCliqueMatches();
          }

          if (show_clique_cross_features) {
            debug_renderer.renderCliqueFeatures();
          }

          if (show_inlier_features) {
            debug_renderer.renderInlierFeatures();
          }
        }
        cv::imshow("debug", debug_renderer.debugImage());
        
        int key = cv::waitKey(debug ? -1 : 1);
        if (key != -1) {
          key &= 0xFF;
        }
        switch(key) {
          case 'f':
            show_features = !show_features;
            break;
          case 'm':
            show_matches = !show_matches;
            break;
          case 'x':
            show_cross_matches = !show_cross_matches;
            break;
          case 'z':
            show_filtered_matches = !show_filtered_matches;
            break;
          case 'c':
            show_clique = !show_clique;
            break;
          case '1':
            show_clique_cross_features = !show_clique_cross_features;
            break;
          case '2':
            show_inlier_features = !show_inlier_features;
            break;
          case 'd':
            debug_renderer.dumpCrossMatches(
                "/tmp/robo-debug/cross-matches-near.yml");
            debug_renderer.dumpClique(
                "/tmp/robo-debug/clique-near.yml");
            break;
          case 27:
            done = true;
            break;
        }
      }
    }
    renderTime.sample(timer.mark());
    fpsMeter.mark();

    // Wait for next frame
    frameTime += frameDt;

    presentTime.sample(timer.mark());

  }

  return 0;
}
