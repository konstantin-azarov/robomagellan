#ifndef __DEBUG_RENDERER__HPP__
#define __DEBUG_RENDERER__HPP__

#include <opencv2/opencv.hpp>
#include <string>

#include "calibration_data.hpp"
#include "frame_processor.hpp"
#include "cross_frame_processor.hpp"
#include "direction_tracker.hpp"

class DebugRenderer {
  public:
    DebugRenderer(
        const FrameProcessor* p1,
        const FrameProcessor* p2,
        const CrossFrameProcessor* cfp,
        const DirectionTracker* direction_tracker, 
        const MonoCalibrationData* mono_calibration,
        int max_width, int max_height);

    bool loop();

    void renderStereo();

    void renderTarget();

    void renderFeatures();

    void renderMatches();

    void renderTargetMatches(
        const std::vector<MonoReprojectionFeature>& features);

    void renderPointFeatures(int p);

    void renderAllCrossMatches();

    void renderFilteredCrossMatches();

    void renderCliqueMatches();

    void renderReprojectionFeatures(
        const std::vector<ReprojectionFeatureWithError>& features);

    void renderCliqueFeatures();

    void renderInlierFeatures();

    void dumpCrossMatches(const std::string& filename);

    void dumpClique(const std::string& filename);
    
    void renderText(const std::string& text);

    void selectKeypoint(int x, int y); 

  private:
    void drawImage_(const cv::Mat& src, const cv::Mat& dest);

    void drawMatch_(const CrossFrameMatch& match);

    void drawCross_(const cv::Point& pt, const cv::Scalar& color);

  private:
    int max_width_, max_height_;

    double scale_;
    int w_, h_;

    const FrameProcessor *p1_, *p2_;
    const CrossFrameProcessor* cfp_;
    const DirectionTracker* dt_;
    const MonoCalibrationData* mono_calibration_;
    
    cv::Mat img_;
    std::pair<int, int> selection_;
};

#endif
