#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <assert.h>

#include "average.hpp"
#include "backward.hpp"
#include "calibration_data.hpp"
#include "clique.hpp"
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

const double CROSS_POINT_DIST_THRESHOLD = 20; // mm

struct CrossFrameMatch {
  CrossFrameMatch() {}

  CrossFrameMatch(const cv::Point3d p1_, const cv::Point3d p2_, int i1_, int i2_) :
    p1(p1_), p2(p2_), i1(i1_), i2(i2_) {
  }

  cv::Point3d p1, p2;
  // Index of the corresponding frame point
  int i1, i2;
};

class CrossFrameProcessor {
  public:
    CrossFrameProcessor(const StereoIntrinsics* intrinsics) 
      : intrinsics_(*intrinsics), reprojection_estimator_(intrinsics) {
    }

    bool process(const FrameProcessor& p1, const FrameProcessor& p2) {
      cout << "Matching" << endl;
      const vector<cv::Point3d>& points1 = p1.points();
      const vector<cv::Point3d>& points2 = p2.points();

      match(p1, p2, matches_[0]);
      match(p2, p1, matches_[1]);

      full_matches_.resize(0);

      for (int i=0; i < matches_[0].size(); ++i) {
        int j = matches_[0][i];
        if (j != -1 && matches_[1][j] == i) {
          full_matches_.push_back(CrossFrameMatch(points1[i], points2[j], i, j));
        }
      }

      buildClique_(p1, p2);

      cout << "Matches = " << full_matches_.size() << "; clique = " << clique_points_[0].size() << endl;

      if (clique_points_[0].size() < 10) {
        return false;
      }

      rigid_estimator_.estimate(clique_points_[0], clique_points_[1]);
      reprojection_estimator_.estimate(reprojection_features_);

      auto rigid_errors = reprojectionError_(
          rigid_estimator_.rot(), rigid_estimator_.t());
      cout << "Reprojection errors rigid ([1->2] [2->1]):" 
        << rigid_errors.first << " " << rigid_errors.second  << endl;

      auto reprojection_errors = reprojectionError_(
          reprojection_estimator_.rot(), reprojection_estimator_.t());
      cout << "Reprojection errors reprojection ([1->2] [2->1]):" 
        << reprojection_errors.first << " " << reprojection_errors.second  << endl;

      cout << "dR = " << rot() << "; dt = " << t() << endl;

      return true;
    }

    const cv::Mat& rot() const { return reprojection_estimator_.rot(); }
    const cv::Point3d& t() const { return reprojection_estimator_.t(); } 

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

          double d1 = descriptorDist(left_descs.first, right_descs.first);
          double d2 = descriptorDist(left_descs.second, right_descs.second);

          double d = max(d1, d2);

          if (d < best_dist) {
            best_dist = d;
            best_j = j;
          }
        }

        matches[i] = best_j;
      }
    }

    void buildClique_(
        const FrameProcessor& p1, 
        const FrameProcessor& p2) {
      // Construct the matrix of the matches
      // matches i and j have an edge is distance between corresponding points in the
      // first image is equal to the distance between corresponding points in the
      // last image.
      int n = full_matches_.size();
      clique_.reset(n);

      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
          auto& m1 = full_matches_[i];
          auto& m2 = full_matches_[j];
          double d1 = norm3(m1.p1 - m2.p1);
          double d2 = norm3(m1.p2 - m2.p2);
          
          if (abs(d1 - d2) < CROSS_POINT_DIST_THRESHOLD) {
            clique_.addEdge(i, j);
          }
        }
      }

      const std::vector<int>& clique = clique_.clique();

      clique_points_[0].resize(clique.size());
      clique_points_[1].resize(clique.size());
      reprojection_features_.resize(clique.size());
      for (int i=0; i < clique.size(); ++i) {
        const auto& m = full_matches_[clique[i]];

        clique_points_[0][i] = m.p1;
        clique_points_[1][i] = m.p2;

        auto& f = reprojection_features_[i];

        f.r1 = m.p1;
        f.r2 = m.p2;
        std::tie(f.s1l, f.s1r) = p1.features(m.i1);
        std::tie(f.s2l, f.s2r) = p2.features(m.i2);
      }
    }

    // Reprojection error 1 -> 2 & 2 -> 1
    // R and tm transform 1 into 2
    std::pair<double, double> reprojectionError_(
        const cv::Mat& R, 
        const cv::Point3d& tm) {
      double res1 = 0, res2 = 0;

      for (auto f : reprojection_features_) {
        auto p1 = projectPoint(intrinsics_, R.t()*(f.r2 - tm));
        auto p2 = projectPoint(intrinsics_, R*f.r1 + tm);

        res1 += norm2(p1.first - f.s1l) + norm2(p1.second - f.s1r);
        res2 += norm2(p2.first - f.s2l) + norm2(p2.second - f.s2r);
      }

      int n = reprojection_features_.size();

      return make_pair(res1 / n, res2 / n);
    }
  
  private:
    const StereoIntrinsics& intrinsics_;

    // matches[0][i] - best match in the second frame for i-th feature in the first frame
    // matches[1][j] - best match in the first frame for j-th feature in the second frame
    vector<int> matches_[2];
    // 3d point matches between frames
    vector<CrossFrameMatch> full_matches_;
    // Clique builder
    Clique clique_;
    // Points corresponding to clique from the first and second frames
    vector<cv::Point3d> clique_points_[2];
    // Original features (or rather their locations) from the first and second frames
    // respectively
    vector<ReprojectionFeature> reprojection_features_;
    // estimators
    RigidEstimator rigid_estimator_;
    ReprojectionEstimator reprojection_estimator_;
};

void onMouse(int event, int x, int y, int flags, void* ptr) {
  const FrameProcessor* p = static_cast<FrameProcessor*>(ptr);

  if (event == cv::EVENT_LBUTTONDOWN) {
    p->printKeypointInfo(x, y);
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
