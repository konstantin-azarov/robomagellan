#ifndef __DEBUG_RENDERER__HPP__
#define __DEBUG_RENDERER__HPP__

#include <opencv2/opencv.hpp>
#include <string>


class FrameProcessor;
class CrossFrameProcessor;

class DebugRenderer {
  public:
    DebugRenderer();

    void start(
        const FrameProcessor* p1, 
        const FrameProcessor* p2,
        const CrossFrameProcessor* cfp);

    void renderFeatures();

    void renderMatches();

    void renderPointFeatures(int p);

    void renderCrossMatches();

    void renderText(const std::string& text);

    cv::Mat& debugImage() { return img_ ; }

    void selectKeypoint(int x, int y); 

  private:
    const FrameProcessor *p1_, *p2_;
    const CrossFrameProcessor* cfp_;
    cv::Mat img_;
    std::pair<int, int> selection_;
};

#endif
