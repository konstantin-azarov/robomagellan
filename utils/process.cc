#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include "raw_video_reader.h"
#include "average.hpp"
#include "fps_meter.hpp"
#include "timer.hpp"

using namespace std;

namespace po = boost::program_options;

const int frame_width = 640;
const int frame_height = 480;

struct CalibData {
  cv::Mat Ml, dl, Mr, dr, R, T;

  static CalibData read(const string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    CalibData res;
    fs["Ml"] >> res.Ml;
    fs["dl"] >> res.dl;
    fs["Mr"] >> res.Mr;
    fs["dr"] >> res.dr;
    fs["R"] >> res.R;
    fs["T"] >> res.T;

    return res;
  }
};

struct UndistortMaps {
  cv::Mat x, y;
};

class FrameProcessor {
  private:
    const int LEFT = 0;
    const int RIGHT = 1;

  public:
    FrameProcessor(const CalibData& calib, int width, int height) {

      cv::stereoRectify(
          calib.Ml, calib.dl, 
          calib.Mr, calib.dr, 
          cv::Size(width, height),
          calib.R, calib.T,
          Rl_, Rr_,
          Pl_, Pr_,
          Q_,
          cv::CALIB_ZERO_DISPARITY,
          0); 

      cout << "Ml = " << calib.Ml << endl << "d = " << calib.dl << endl;

      cv::initUndistortRectifyMap(
          calib.Ml, calib.dl, Rl_, Pl_, 
          cv::Size(width, height),
          CV_16SC2,
          maps_[LEFT].x, maps_[LEFT].y );

      cv::initUndistortRectifyMap(
          calib.Mr, calib.dr, Rr_, Pr_, 
          cv::Size(width, height),
          CV_16SC2,
          maps_[RIGHT].x, maps_[RIGHT].y);
    }

    void process(const cv::Mat src[], cv::Mat& debug) {
      for (int i=0; i < 2; ++i) {
        cv::remap(
            src[i], 
            undistorted_image_[i], 
            maps_[i].x, 
            maps_[i].y, 
            cv::INTER_LINEAR);
      }

      hconcat(undistorted_image_[0], undistorted_image_[1], debug);

      cv::Surf surf;
      
    }

  private:
    cv::Mat Rl_, Pl_, Rr_, Pr_, Q_;

    UndistortMaps maps_[2];

    cv::Mat undistorted_image_[2];

    vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
};


int main(int argc, char** argv) {
  string video_file, calib_file;
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

  desc.add_options()
      ("calib-file",
       po::value<string>(&calib_file)->default_value("data/calib.yml"),
       "path to the calibration file");
 
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  
  CalibData calib = CalibData::read(calib_file);

  FrameProcessor processor(calib, frame_width, frame_height);

  RawVideoReader rdr(video_file, frame_width*2, frame_height);
  uint8_t frame_data[frame_width*2*frame_height];
  cv::Mat frame_mat(frame_height, frame_width*2, CV_8UC1, frame_data);
  cv::Mat mono_frames[] = {
    frame_mat(cv::Range::all(), cv::Range(0, frame_width)),
    frame_mat(cv::Range::all(), cv::Range(frame_width, frame_width*2))
  };
  cv::Mat debug_mat;
  cv::Mat render_mat(frame_height, frame_width*2, CV_8UC2);

  cv::namedWindow("video");

  Timer timer;
  Average readTime(fps), 
          processTime(fps),
          renderTime(fps), 
          presentTime(fps);
  FpsMeter fpsMeter(fps);

  double frameTime = nanoTime();
  double frameDt = 1.0/fps;

  bool done = false;
  while (!done) {
    timer.mark();

    if (!rdr.nextFrame(frame_data)) {
      break;
    }

    readTime.sample(timer.mark());

    processor.process(mono_frames, debug_mat);

    processTime.sample(timer.mark());


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

    cv::cvtColor(debug_mat, render_mat, CV_GRAY2RGB);
    cv::putText(
        render_mat, 
        buf, 
        cv::Point(0, frame_height-10), 
        cv::FONT_HERSHEY_PLAIN, 
        1.0, 
        cv::Scalar(0, 255, 0));

    cv::imshow("video", render_mat);

    renderTime.sample(timer.mark());
    fpsMeter.mark();

    // Wait for next frame
    frameTime += frameDt;
    long wait = max(1, static_cast<int>((frameTime - nanoTime())*1000));

    int key = cv::waitKey(wait);
    if (key != -1) {
      key &= 0xFF;
    }
    switch(key) {
      case 27:
        done = true;
        break;
    }

    presentTime.sample(timer.mark());
  }

  return 0;
}
