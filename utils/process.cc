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

#include "frame_processor.inl"

const double CROSS_POINT_DIST_THRESHOLD = 20; // mm

class CrossFrameProcessor {
  public:
    CrossFrameProcessor() {
    }

    void process(const FrameProcessor& p1, const FrameProcessor& p2) {
      cout << "Matching" << endl;
      const vector<cv::Point3d>& points1 = p1.points();
      const vector<cv::Point3d>& points2 = p2.points();

      match(p1, p2, matches_[0]);
      match(p2, p1, matches_[1]);

      full_matches_.resize(0);

      for (int i=0; i < matches_[0].size(); ++i) {
        int j = matches_[0][i];
        if (j != -1 && matches_[1][j] == i) {
          full_matches_.push_back(std::make_pair(points1[i], points2[j]));
        }
      }

      cout << "Cross matches: " << full_matches_.size() << endl;


      buildMatchesGraph_();
      buildClique_();
    }

  private:
    void match(
        const FrameProcessor& p1, 
        const FrameProcessor& p2,
        vector<int>& matches) {
      const vector<cv::Point3d>& points1 = p1.points();
      const vector<cv::Point3d>& points2 = p2.points();

      matches.resize(points1.size());

      for (int i = 0; i < points1.size(); ++i) {
        int best_j = -1;
        double best_dist = 1E+15;
        auto left_descs = p1.pointDescriptors(i);
        for (int j = 0; j < points2.size(); ++j) {
          auto right_descs = p2.pointDescriptors(j);

          double d1 = cv::norm(left_descs.first, right_descs.first);
          double d2 = cv::norm(left_descs.second, right_descs.second);

          double d = max(d1, d2);

          if (d < best_dist) {
            best_dist = d;
            best_j = j;
          }
        }

        matches[i] = best_j;
      }
    }

    void buildMatchesGraph_() {
      // Construct the matrix of the matches
      // matches i and j have an edge is distance between corresponding points in the
      // first image is equal to the distance between corresponding points in the
      // last image.
      int n = full_matches_.size();
      matrix_.resize(n*n);
      degrees_.resize(n);

      for (int i = 0; i < n; ++i) {
        matrix_[i*n] = 1;
        for (int j = 0; j < i; ++j) {
          auto& pts1 = full_matches_[i];
          auto& pts2 = full_matches_[j];
          double d1 = cv::norm(pts1.first - pts2.first);
          double d2 = cv::norm(pts1.second - pts2.second);
          
          if (abs(d1 - d2) < CROSS_POINT_DIST_THRESHOLD) {
            matrix_[i*n + j] = matrix_[j*n + i] = 1;
            degrees_[i]++;
            degrees_[j]++;
          } else {
            matrix_[i*n + j] = matrix_[j*n + i] = 0;
          }
        }
      }
    }

    void buildClique_() {
      int n = full_matches_.size();

      // Estimate max clique
      // find vertex with maximum degree
      clique_.resize(0);

      int best = 0;
      for (int i = 1; i < n; ++i) {
        if (degrees_[i] > degrees_[best]) {
          best = i;
        }
      }

      candidates_[0].resize(0);
      for (int i=0; i < n; ++i) {
        if (i != best) {
          candidates_[0].push_back(i);
        }
      }

      int t = 0;
      while (best >= 0) {
        clique_.push_back(best);
      
        candidates_[1-t].resize(0);
        for (int i=0; i < candidates_[t].size(); ++i) {
          if (best != candidates_[t][i] && matrix_[best*n + candidates_[t][i]]) {
            candidates_[1-t].push_back(candidates_[t][i]);
          }
        }

        best = -1;
        for (int i=0; i < candidates_[1-t].size(); ++i) {
          if (best == -1 || degrees_[candidates_[1-t][i]] > degrees_[best]) {
           best = candidates_[1-t][i];
          }
        }

        t = 1-t;
      }

      cout << "Clique size: " << clique_.size() << endl;
    }

  private:
    vector<int> matches_[2];
    vector<std::pair<cv::Point3d, cv::Point3d> > full_matches_;
    vector<int> matrix_;
    vector<int> degrees_;
    vector<int> clique_;
    vector<int> candidates_[2];
};

void onMouse(int event, int x, int y, int flags, void* ptr) {
  const FrameProcessor* p = static_cast<FrameProcessor*>(ptr);

  if (event == cv::EVENT_LBUTTONDOWN) {
    p->printKeypointInfo(x, y);
  }
}

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
  
  CalibData calib = CalibData::read(calib_file);

  FrameProcessor frame_processors[] = {
    FrameProcessor(calib, frame_width, frame_height),
    FrameProcessor(calib, frame_width, frame_height)
  };
  CrossFrameProcessor cross_processor;

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
      cross_processor.process(
          frame_processors[!(frame_index & 1)],
          processor);
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
    long wait = max(1, static_cast<int>((frameTime - nanoTime())*1000));

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
