#include <libuvc/libuvc.h>
#include <iostream>

using namespace std;

#include "camera.hpp"

#include "frame_buffer_queue.hpp"

#define HWREV 0x5102
#define FWREV 0x01bf

#define VENDOR_ID 0x2b03//0x2a0b
#define PRODUCT_ID 0xf580//0x00f5

#define CAMERA_DEBUG

Camera::Camera() {
  ctx_ = 0;
  dev_ = 0;
  devh_ = 0;
}

Camera::~Camera() {

}

bool Camera::init(int width, int height, int fps) {
  frame_width_ = width;
  frame_height_ = height;
  frame_counter_ = 0;
  tmp_buffer_ = new uint8_t[width*height*2];

  queue_ = new FrameBufferQueue(width*height*2, 30*10);

  uvc_stream_ctrl_t ctrl;
  uvc_error_t res;
  /* Initialize a UVC service context. Libuvc will set up its own libusb
   * context. Replace NULL with a libusb_context pointer to run libuvc
   * from an existing libusb context. */
  res = uvc_init(&ctx_, NULL);
  if (res < 0) {
    uvc_perror(res, "uvc_init");
    return false;
  }
  /* Locates the first attached UVC device, stores in dev */
  res = uvc_find_device(
      ctx_, &dev_,
      VENDOR_ID, PRODUCT_ID, NULL); /* filter devices: vendor_id, product_id, "serial_num" */
  if (res < 0) {
    uvc_perror(res, "uvc_find_device"); /* no devices found */
    return false;
  }

  /* Try to open the device: requires exclusive access */
  res = uvc_open(dev_, &devh_);
  if (res < 0) {
    uvc_perror(res, "uvc_open"); /* unable to open device */
    return false;
  }

#ifdef CAMERA_DEBUG
  /* Print out a message containing all the information that libuvc
   * knows about the device */
  uvc_print_diag(devh_, stderr);
#endif

  const uvc_extension_unit_t* unit = uvc_get_extension_units(devh_);

#ifdef CAMERA_DEBUG
  while (unit != 0) {
    printf("Extension unit %d:\n", unit->bUnitID);
    printf("     ");
    for (int i=0; i < 16; ++i) {
      printf("%02x", unit->guidExtensionCode[i]);
    }
    printf("\n");
    unit = unit->next;
  }
#endif

//  uint8_t unitId = 3;
//  uint8_t ctl = 0x07;
//  uint8_t buf[36 + 4 + 9];
//
//  uint8_t ret = uvc_get_ctrl(devh_, unitId, ctl, buf, sizeof(buf), UVC_GET_CUR);
//
//  uint16_t hw_rev = buf[0] | ((uint16_t)buf[1] << 8);
//  uint16_t fw_rev = buf[0] | buf[2] | ((uint16_t)buf[3] << 8);
//
//  if (hw_rev != HWREV || fw_rev != FWREV) {
//    cerr << "Does not seem to be the right camera" << hw_rev << " " << fw_rev << endl;
//    return false;
//  }

  /* Try to negotiate a 640x480 30 fps YUYV stream profile */
  res = uvc_get_stream_ctrl_format_size(
      devh_, &ctrl, /* result stored in ctrl */
      UVC_FRAME_FORMAT_YUYV, /* YUV 422, aka YUV 4:2:2. try _COMPRESSED */
      width, height, fps
  );
#ifdef CAMERA_DEBUG
  /* Print out the result */
  uvc_print_stream_ctrl(&ctrl, stderr);
#endif

  if (res < 0) {
    uvc_perror(res, "get_mode"); /* device doesn't provide a matching stream */
    return false;
  }

  /* Start the video stream. The library will call user function cb:
   *   cb(frame, (void*) 12345)
   */
  res = uvc_start_streaming(devh_, &ctrl, &Camera::staticCallback, this, 0);
  if (res < 0) {
    uvc_perror(res, "start_streaming"); /* unable to start stream */
    return false;
  }

  /* std::cout << "SetAE = " << uvc_strerror(uvc_set_ae_mode(devh_, 2)) << std::endl; /1* e.g., turn on auto exposure *1/ */
  /* std::cout << "SetAEP = " << uvc_strerror(uvc_set_ae_priority(devh_, 0)) << std::endl; */
  
  /* uint8_t ae_mode; */
  /* auto err = uvc_get_ae_mode(devh_, &ae_mode, UVC_GET_CUR); */
  /* std::cout << "AE = " << err << " xcxc " << (int)ae_mode << std::endl; */

  return true;
}

#define PARAM_ACCESSOR(type, name, uvc_func) \
  type Camera::get##name() { \
    type res; \
    auto err = uvc_get_##uvc_func(devh_, &res, UVC_GET_CUR); \
    if (err != 0) { \
      std::cout << "Cannot get " #name << ": " << uvc_strerror(err) << std::endl; \
      return -1; \
    } \
    return res; \
  } \
\
  void Camera::get##name##Limits(type& minV, type& maxV) { \
    if (uvc_get_##uvc_func(devh_, &minV, UVC_GET_MIN) < 0) \
      minV = -1; \
    if (uvc_get_##uvc_func(devh_, &maxV, UVC_GET_MAX) < 0) \
      maxV = -1; \
  } \
\
  void Camera::set##name(type value) { \
    auto err = uvc_set_##uvc_func(devh_, value); \
    if (err != 0) { \
      std::cout << "Setting " #name " failed: " << uvc_strerror(err) << std::endl; \
    } \
  } 

PARAM_ACCESSOR(uint32_t, Exposure, exposure_abs);
PARAM_ACCESSOR(uint16_t, Iris, iris_abs);
PARAM_ACCESSOR(uint16_t, Focus, focus_abs);
PARAM_ACCESSOR(uint16_t, Gain, gain);

#undef PARAM_ACCESSOR

void Camera::shutdown() {
  /* End the stream. Blocks until last callback is serviced */
  uvc_stop_streaming(devh_);

  /* Release our handle on the device */
  uvc_close(devh_);

  /* Release the device descriptor */
  uvc_unref_device(dev_);
  /* Close the UVC context. This closes and cleans up any existing device handles,
   * and it closes the libusb context if one was not provided. */
  uvc_exit(ctx_);

  delete queue_;
  delete tmp_buffer_;
}

void Camera::nextFrame(uint8_t* frame) {
  queue_->nextFrame(frame);
}

void Camera::staticCallback(uvc_frame_t* frame, void* ptr) {
  static_cast<Camera*>(ptr)->callback(frame);
}

void Camera::callback(uvc_frame_t* frame) {
  uint8_t* data = static_cast<uint8_t*>(frame->data);
  uint8_t* left_data = tmp_buffer_;
  uint8_t* right_data = tmp_buffer_ + frame_width_;

  frame_counter_++;
//  if (frame_counter_ % 2 == 0) {
//    return;
//  }

  if (frame->data_bytes != frame_width_*frame_height_*2) {
    cerr << "Incomplete frame: " << frame->sequence << " " << frame->data_bytes << endl;
    return;
  }

//  for (int i = 0; i < frame_height_; ++i) {
//    for (int j = 0; j < frame_width_; ++j) {
//      *(left_data++) = *(data++);
//      *(right_data++) = *(data++);
//    }
//    left_data += frame_width_;
//    right_data += frame_width_;
//  }


  if (frame_counter_ != frame->sequence) {
    cerr << "Missed frame(s): "
         << frame_counter_ << " " << frame->sequence << endl;
    frame_counter_ = frame->sequence;
  }

  queue_->addFrame(data);
}
