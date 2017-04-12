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


  return true;
}

void Camera::setExposure(int v) {
  uint8_t cmd[64] = { 0x07, 0x24, 0x01, 0x00, (v >> 8) & 0xFF, v & 0xFF, 0x00 };

  uvc_set_ctrl(devh_, 0x03, 0x02, cmd, sizeof(cmd));

  cmd[2] = 0x02;
  uvc_set_ctrl(devh_, 0x03, 0x02, cmd, sizeof(cmd));
}

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

  if (frame->data_bytes != frame_width_*frame_height_*2) {
    cerr << "Incomplete frame: " << frame->sequence << " " << frame->data_bytes << endl;
    return;
  }

  if (frame_counter_ != frame->sequence) {
    cerr << "Missed frame(s): "
         << frame_counter_ << " " << frame->sequence << endl;
    frame_counter_ = frame->sequence;
  }

  queue_->addFrame(data);
}
