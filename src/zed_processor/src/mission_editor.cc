#include <boost/format.hpp>
#include <boost/program_options.hpp> 
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <sstream>
#include <string>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "calibration_data.hpp"
#include "direction_target.hpp"

namespace po = boost::program_options;
using boost::format;

struct MissionEditorException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

cv::Size parseResolution(const std::string& resolution_str) {
  if (resolution_str == "1280x720") {
    return cv::Size(1280, 720);
  }

  throw new MissionEditorException("Unsupported resolution: " + resolution_str);
}

backward::SignalHandling sh;

enum class SelectionState {
  NONE,
  FEATURE_RECT_1,
  FEATURE_RECT_2,
  TARGET_DIRECTION
};

struct State {
  SelectionState selection_state = SelectionState::NONE;

  cv::Point2i direction = cv::Point2i(-1, -1);
  cv::Point2i features_tl = cv::Point2i(-1, -1);
  cv::Point2i features_br = cv::Point2i(-1, -1);

  cv::Point2i mouse_pos = cv::Point2i(0, 0);
  
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

void onMouse(int event, int x, int y, int flags, void* ptr) {
  State* state = reinterpret_cast<State*>(ptr);

  if (event == cv::EVENT_MOUSEMOVE) {
    state->mouse_pos = cv::Point2i(x, y);
  } else if (event == cv::EVENT_LBUTTONDOWN) {
    switch (state->selection_state) {
      case SelectionState::NONE:
        break;
      case SelectionState::FEATURE_RECT_1:
        state->features_tl = cv::Point2i(x, y);
        state->selection_state = SelectionState::FEATURE_RECT_2;
        break;
      case SelectionState::FEATURE_RECT_2:
        state->features_br = cv::Point2i(x, y);
        state->selection_state = SelectionState::NONE;
        break;
      case SelectionState::TARGET_DIRECTION:
        state->direction = cv::Point2i(x, y);
        state->selection_state = SelectionState::NONE;
        break;
    }
  }
}

void save(
    const State& state, 
    const std::string& image_file,
    const std::string& filename) {
  DirectionTarget target;

  target.image_file = image_file;
  target.target = state.direction;

  std::vector<int> matches;
  for (int i = 0; i < state.keypoints.size(); ++i) {
    const auto& kp = state.keypoints[i];
    if (kp.pt.x >= state.features_tl.x && kp.pt.x <= state.features_br.x &&
        kp.pt.y >= state.features_tl.y && kp.pt.y <= state.features_br.y) {
      matches.push_back(i);
    }
  }

  target.keypoints.resize(matches.size());
  target.descriptors.create(matches.size(), 512/8, CV_8UC1);

  for (int i = 0; i < matches.size(); ++i) {
    target.keypoints[i] = state.keypoints[matches[i]].pt;
    state.descriptors.row(matches[i]).copyTo(target.descriptors.row(i));
  }

  target.write(filename);

/*   auto t2 = target.read(filename); */
/*   std::cout << (t2.image_file == target.image_file); */
/*   std::cout << (t2.target == target.target); */
/*   std::cout << !cv::countNonZero(t2.descriptors != target.descriptors); */
/*   std::cout << (t2.keypoints == target.keypoints); */
/*   std::cout << std::endl; */
}

int main(int argc, char** argv) {
  std::string stereo_calib_file, calib_file, image_file;
  std::string target_resolution_str;
  std::string plan_file;

  po::options_description desc("Mission editor");
  desc.add_options()
      ("stereo-calib-file",
       po::value(&stereo_calib_file)->default_value("utils/data/calib.yml"),
       "stereo camera calibration file")
      ("calib-file",
       po::value(&calib_file)->default_value("utils/data/calib_phone.yml"),
       "camera calibration file")
      ("target-resolution",
       po::value(&target_resolution_str)->default_value("1280x720"),
       "convert images to this resolution for feature detection")
      ("image",
       po::value(&image_file)->required(),
       "Image to process")
      ("plan-file",
       po::value(&plan_file)->required(),
       "Where to save the plan");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  StereoCalibrationData stereo_calib(
      RawStereoCalibrationData::read(stereo_calib_file));

  MonoCalibrationData calib(
      RawMonoCalibrationData::read(calib_file),
      stereo_calib.Pl.colRange(0, 3),
      parseResolution(target_resolution_str));

  auto raw_image = cv::imread(image_file, cv::IMREAD_GRAYSCALE);
  cv::Mat undistorted_image;
  cv::remap(
      raw_image, 
      undistorted_image, 
      calib.undistort_maps.x,
      calib.undistort_maps.y,
      cv::INTER_LINEAR);

  int threshold = 35;

  cv::Mat debug_img;
  State state;

  cv::namedWindow("editor");
  cv::setMouseCallback("editor", onMouse, &state);

  auto freak = cv::xfeatures2d::FREAK::create(true, false);

  bool done = false;
  while (!done) {
    cv::FAST(
        undistorted_image,
        state.keypoints, 
        threshold,
        true);

    freak->compute(undistorted_image, state.keypoints, state.descriptors);

    cv::cvtColor(undistorted_image, debug_img, CV_GRAY2RGB);

    for (const auto& kp : state.keypoints) {
      cv::circle(debug_img, kp.pt, 2, cv::Scalar(0, 255, 0), 1.0);
    }

    std::ostringstream text;

    text 
      << "threshold = " << threshold
      << "; mouse = [" << state.mouse_pos.x << ", " << state.mouse_pos.y << "]";

    if (state.direction.x != -1) {
      cv::line(
          debug_img, 
          state.direction - cv::Point(10, 10),
          state.direction + cv::Point(10, 10),
          cv::Scalar(0, 0, 255),
          2);

      cv::line(
          debug_img, 
          state.direction - cv::Point(-10, 10),
          state.direction + cv::Point(-10, 10),
          cv::Scalar(0, 0, 255),
          2);
    }

    if (state.features_br.x != -1 || 
        state.selection_state == SelectionState::FEATURE_RECT_2) {
      cv::Point br = state.selection_state == SelectionState::FEATURE_RECT_2 
          ? state.mouse_pos
          : state.features_br;

      cv::rectangle(
          debug_img, 
          state.features_tl, 
          br,
          cv::Scalar(0, 0, 255),
          state.selection_state == SelectionState::FEATURE_RECT_2 ? 1 : 2);
    }

    switch (state.selection_state) {
      case SelectionState::NONE:
        break;
      case SelectionState::FEATURE_RECT_1:
        text << "; Select first corner";
        break;
      case SelectionState::FEATURE_RECT_2:
        text << "; Select second corner";
        break;
      case SelectionState::TARGET_DIRECTION:
        text << "; Select target direction";
    }

    cv::putText(
        debug_img, 
        text.str(),
        cv::Point(0, 720 - 20),
        cv::FONT_HERSHEY_PLAIN,
        1.0,
        cv::Scalar(0, 0, 255),
        2.0,
        cv::LINE_AA);

    cv::imshow("editor", debug_img);
    int key = cv::waitKey(25);
    if (key != -1) {
      key &= 0xFF;
    }
    switch (key) {
      case 27:
        done = true;
        break;
      case '=':
      case '+':
        threshold++;
        break;
      case '-':
        threshold--;
        break;
      case 'd':
        state.selection_state = SelectionState::TARGET_DIRECTION;
        state.direction = cv::Point(-1, -1);
        break;
      case 'f':
        state.selection_state = SelectionState::FEATURE_RECT_1;
        state.features_br = cv::Point(-1, -1);
        break;
      case 's':
        save(state, image_file, plan_file);
        break;
      case -1:
        break;
      default:
        std::cout << "Unknown key code: " << key << std::endl;
    }
  }

}
