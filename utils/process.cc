#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <assert.h>

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

struct Match {
  int leftIndex, rightIndex;
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

      cout << "Ml = " << calib.Ml << endl << "dl = " << calib.dl << endl;
      cout << "Mr = " << calib.Mr << endl << "dr = " << calib.dr << endl;

      cv::initUndistortRectifyMap(
          calib.Ml, calib.dl, Rl_, Pl_, 
          cv::Size(width, height),
          CV_32FC1,
          maps_[LEFT].x, maps_[LEFT].y);

      cv::initUndistortRectifyMap(
          calib.Mr, calib.dr, Rr_, Pr_, 
          cv::Size(width, height),
          CV_32FC1,
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

        cv::SURF surf;
        surf(undistorted_image_[i], cv::noArray(), keypoints_[i], descriptors_[i]);

        order_[i].resize(keypoints_[i].size());
        for (int j=0; j < order_[i].size(); ++j) {
          order_[i][j] = j;
        }
        
        sort(order_[i].begin(), order_[i].end(), [this, i](int a, int b) -> bool {
            auto& k1 = keypoints_[i][a].pt;
            auto& k2 = keypoints_[i][b].pt;

            return (k1.y < k2.y || (k1.y == k2.y && k1.x < k2.x));
        });
      }

      for (int i=0; i < 2; ++i) {
        int j = 1 - i;
        match(
            keypoints_[i], order_[i], descriptors_[i],
            keypoints_[j], order_[j], descriptors_[j],
            matches_[i]);
      }

      points_.resize(0);
      for (int i=0; i < matches_[0].size(); ++i) {
        int j = matches_[0][i];
        if (j != -1 && matches_[1][j] == i) {
          auto& kp1 = keypoints_[0][i].pt;
          auto& kp2 = keypoints_[1][j].pt;
          points_.push_back(cv::Point3d(kp1.x, (kp1.y + kp2.y)/2, kp1.x - kp2.x));
        }
      }

      cout << "Matches = " << points_ << endl;
      cv::perspectiveTransform(points_, points_, Q_);

      drawDebugImage(debug);
    }

    void match(const vector<cv::KeyPoint>& kps1,
               const vector<int>& idx1,
               const cv::Mat& desc1,
               const vector<cv::KeyPoint>& kps2,
               const vector<int>& idx2,
               const cv::Mat& desc2,
               vector<int>& matches) {
      matches.resize(kps1.size());

      int j0 = 0, j1 = 0;

      for (int i : idx1) {
        auto& pt1 = kps1[i].pt;

        matches[i] = -1;

        while (j0 < kps2.size() && kps2[idx2[j0]].pt.y < pt1.y - 2)
          ++j0;

        while (j1 < kps2.size() && kps2[idx2[j1]].pt.y < pt1.y + 2)
          ++j1;

//        cout << kps2.size() << " " << j0 << " " << j1 << " " << pt1 << " " << kps2[idx2[j0]].pt << " " << kps2[idx2[j1]].pt << endl;

        assert(j1 >= j0);

        double best_d = 1E+15, second_d = 1E+15;
        double best_j = -1;

        for (int jj = j0; jj < j1; jj++) {
          int j = idx2[jj];
          auto& pt2 = kps2[j].pt;

          assert(abs(pt1.y - pt2.y) < 2);

          if (abs(pt1.x - pt2.x) < 100) {
            double dist = cv::norm(desc1.row(i) - desc2.row(j));
            if (dist < best_d) {
              best_d = dist;
              best_j = j;
            } else if (dist < second_d) {
              second_d = dist;
            }
          }
        }
        
        if (best_j > -1 && best_d / second_d < 0.8) {
          matches[i] = best_j;
        }
      }
    }

    void drawDebugImage(cv::Mat& debug) {
      int w = undistorted_image_[0].cols;
      int h = undistorted_image_[0].rows;
    
      debug.create(h, 2*w, CV_8UC3);

      cv::cvtColor(
          undistorted_image_[0], 
          debug(cv::Range::all(), cv::Range(0, w)), 
          CV_GRAY2RGB);
      cv::cvtColor(
          undistorted_image_[1], 
          debug(cv::Range::all(), cv::Range(w, 2*w)), 
          CV_GRAY2RGB);

      for (int t = 0; t < 2; ++t) {
        for (int i = 0; i < keypoints_[t].size(); ++i) {
          auto& pt = keypoints_[t][i].pt;

          int c = matches_[t][i];
          bool mutual = c != -1 && matches_[1-t][c] == i;
          if (mutual) {
            cv::circle(debug, cv::Point(pt.x + w*t, pt.y), 3, cv::Scalar(0, 255, 0));
          } else {
            cv::circle(debug, cv::Point(pt.x + w*t, pt.y), 3, cv::Scalar(0, 0, 255));
          }

          if (c != -1) {
            auto& pt2 = keypoints_[1-t][c].pt;
            if (!mutual || t == 0) {
              cv::line(
                  debug, 
                  cv::Point(pt.x + w*t, pt.y), 
                  cv::Point(pt2.x + w*(1-t), pt2.y),
                  cv::Scalar(0, mutual*255, (!mutual)*255));
            }
          }
        }
      }
    }

    void printKeypointInfo(int x, int y) const {
      cout << "debug x = " << x << " y = " << y << endl;

      int w = undistorted_image_[0].cols;
      int t = x >= w ? 1 : 0;
      x %= w;
    
      int best = -1;
      double bestD = 1E+15;


      for (int i = 0; i < keypoints_[t].size(); ++i) {
        auto& pt = keypoints_[t][i].pt;
        double d = sqrt((pt.x - x)*(pt.x - x) + (pt.y - y)*(pt.y - y));

        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }

      if (best != -1) {
        auto& pt = keypoints_[t][best].pt;
        cout << "i = " << best << " x = " << pt.x << " y = " << pt.y << endl;
      }
    }

  private:
    cv::Mat Rl_, Pl_, Rr_, Pr_, Q_;

    UndistortMaps maps_[2];

    cv::Mat undistorted_image_[2];

    vector<cv::KeyPoint> keypoints_[2];
    cv::Mat descriptors_[2];
    vector<int> order_[2];
    vector<int> matches_[2]; 
    vector<cv::Point3d> points_;
};

void onMouse(int event, int x, int y, int flags, void* ptr) {
  const FrameProcessor* p = static_cast<FrameProcessor*>(ptr);

  if (event == cv::EVENT_LBUTTONDOWN) {
    p->printKeypointInfo(x, y);
  }
}


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
  cv::Mat debug_mat(frame_height, frame_width*2, CV_8UC2);

  cv::namedWindow("video");
  cv::setMouseCallback("video", onMouse, &processor);

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
    long wait = max(1, static_cast<int>((frameTime - nanoTime())*1000));

    int key = cv::waitKey(0);
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