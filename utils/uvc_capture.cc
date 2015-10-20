#include "libuvc/libuvc.h"
#include "boost/format.hpp"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <string>
#include <iostream>

using namespace std;
using boost::format;
using boost::str;

pthread_mutex_t frame_lock = PTHREAD_MUTEX_INITIALIZER;

cv::Mat last_frame_left(480, 640, CV_8UC1),
        last_frame_right(480, 640, CV_8UC1);

int frame_counter = 0;

void cb(uvc_frame_t *frame, void *ptr) {
  uint8_t* data = static_cast<uint8_t*>(frame->data);
  uint8_t* left_data = last_frame_left.data;
  uint8_t* right_data = last_frame_right.data;

  pthread_mutex_lock(&frame_lock);
  for (int i=0; i < frame->data_bytes >> 1; ++i) {
    *(left_data++) = *(data++);
    *(right_data++) = *(data++);
  }
  frame_counter++;
  pthread_mutex_unlock(&frame_lock);
}

int main(int argc, char **argv) {
  uvc_context_t *ctx;
  uvc_device_t *dev;
  uvc_device_handle_t *devh;
  uvc_stream_ctrl_t ctrl;
  uvc_error_t res;
  /* Initialize a UVC service context. Libuvc will set up its own libusb
   * context. Replace NULL with a libusb_context pointer to run libuvc
   * from an existing libusb context. */
  res = uvc_init(&ctx, NULL);
  if (res < 0) {
    uvc_perror(res, "uvc_init");
    return res;
  }
  puts("UVC initialized");
  /* Locates the first attached UVC device, stores in dev */
  res = uvc_find_device(
      ctx, &dev,
      0, 0, NULL); /* filter devices: vendor_id, product_id, "serial_num" */
  if (res < 0) {
    uvc_perror(res, "uvc_find_device"); /* no devices found */
  } else {
    puts("Device found");
    /* Try to open the device: requires exclusive access */
    res = uvc_open(dev, &devh);
    if (res < 0) {
      uvc_perror(res, "uvc_open"); /* unable to open device */
    } else {
      puts("Device opened");
      /* Print out a message containing all the information that libuvc
       * knows about the device */
      uvc_print_diag(devh, stderr);


      const uvc_extension_unit_t* unit = uvc_get_extension_units(devh);
      while (unit != 0) {
        printf("Extension unit %d:\n", unit->bUnitID);
        printf("     ");
        for (int i=0; i < 16; ++i) {
          printf("%02x", unit->guidExtensionCode[i]);
        }
        printf("\n");
        unit = unit->next;
      }

      uint8_t unitId = 3;
      uint8_t ctl = 0x07;
      uint8_t buf[36 + 4 + 9];

      uint8_t ret = uvc_get_ctrl(devh, unitId, ctl, buf, sizeof(buf), UVC_GET_CUR);
      printf("Got %d bytes\n", ret);

      uint16_t hw_rev = buf[0] | ((uint16_t)buf[1] << 8);
      uint16_t fw_rev = buf[0] | buf[2] | ((uint16_t)buf[3] << 8);

      printf("hwrev = 0x%04x, fwrev = 0x%04x\n", hw_rev, fw_rev);

      /* Try to negotiate a 640x480 30 fps YUYV stream profile */
      res = uvc_get_stream_ctrl_format_size(
          devh, &ctrl, /* result stored in ctrl */
          UVC_FRAME_FORMAT_YUYV, /* YUV 422, aka YUV 4:2:2. try _COMPRESSED */
          640, 480, 30 /* width, height, fps */
      );
      /* Print out the result */
      uvc_print_stream_ctrl(&ctrl, stderr);
      if (res < 0) {
        uvc_perror(res, "get_mode"); /* device doesn't provide a matching stream */
      } else {
        /* Start the video stream. The library will call user function cb:
         *   cb(frame, (void*) 12345)
         */
        res = uvc_start_streaming(devh, &ctrl, cb, 0, 0);
        if (res < 0) {
          uvc_perror(res, "start_streaming"); /* unable to start stream */
        } else {
          puts("Streaming...");
          cv::namedWindow("preview");

          uvc_set_ae_mode(devh, 1); /* e.g., turn on auto exposure */

          cv::Mat combined;
          int current_snapshot_index = 0;
          bool done = false;
          while(!done) {
            pthread_mutex_lock(&frame_lock);
            cv::hconcat(last_frame_left, last_frame_right, combined);
            pthread_mutex_unlock(&frame_lock);
            cv::imshow("preview", combined);

            int key = cv::waitKey(40);
            switch (key) {
              case 27:
                done = true;
                break;
              case 32:
                printf("Snapshot %d!\n", ++current_snapshot_index);
                pthread_mutex_lock(&frame_lock);

                cv::imwrite(str(format("snapshots/left_%d.png") % current_snapshot_index), last_frame_left);
                cv::imwrite(str(format("snapshots/right_%d.png") % current_snapshot_index), last_frame_right);

                pthread_mutex_unlock(&frame_lock);
                break;
              case -1:
                break;
              default:
                printf("Unknown code %d\n", key);
                break;
            }
          }

          /* End the stream. Blocks until last callback is serviced */
          uvc_stop_streaming(devh);
          puts("Done streaming.");
        }
      }
      /* Release our handle on the device */
      uvc_close(devh);
      puts("Device closed");
    }
    /* Release the device descriptor */
    uvc_unref_device(dev);
  }
  /* Close the UVC context. This closes and cleans up any existing device handles,
   * and it closes the libusb context if one was not provided. */
  uvc_exit(ctx);
  puts("UVC exited");
  return 0;
}
