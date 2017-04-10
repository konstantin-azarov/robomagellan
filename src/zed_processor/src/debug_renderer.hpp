#ifndef __DEBUG_RENDERER__HPP__
#define __DEBUG_RENDERER__HPP__

#include <opencv2/opencv.hpp>
#include <string>

#include "calibration_data.hpp"
#include "frame_processor.hpp"
#include "cross_frame_processor.hpp"
#include "direction_tracker.hpp"

struct CrossFrameMatch;

class DebugRenderer {
  public:
    static DebugRenderer* create(
        const StereoCalibrationData& calib,
        const FrameData& f,
        const FrameDebugData& fd_prev,
        const FrameDebugData& fd_cur, 
        const CrossFrameDebugData& cfd,
        const Eigen::Affine3d* ground_truth_t, 
        int max_width, int max_height);

    virtual bool loop() = 0;
};

#endif
