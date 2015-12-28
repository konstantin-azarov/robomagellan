#ifndef __CAMERA_HPP__
#define __CAMERA_HPP__

#include <stdint.h>
#include <libuvc/libuvc.h>

class FrameBufferQueue;

class Camera {
  public:
    Camera();
    ~Camera();

    bool init(int width, int height, int fps);
    void shutdown();

    // Blocking, copies the next frame (side-by-side) to the data array of size
    // at least width*height*2
    void nextFrame(uint8_t* data);

  private:
    static void staticCallback(uvc_frame_t* frame, void* ptr);
    void callback(uvc_frame_t* frame);

  private:
    uvc_context_t *ctx_;
    uvc_device_t *dev_;
    uvc_device_handle_t *devh_;

    FrameBufferQueue* queue_;
    uint8_t* tmp_buffer_;
    int frame_counter_;

    int frame_width_, frame_height_;
};

#endif
