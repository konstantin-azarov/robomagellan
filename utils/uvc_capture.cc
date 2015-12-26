#include <libuvc/libuvc.h>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <string>
#include <iostream>

#include "frame_buffer_queue.hpp"

using namespace std;
using boost::format;
using boost::str;

namespace po = boost::program_options;

const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
const int FRAME_SIZE = FRAME_WIDTH*FRAME_HEIGHT;

FrameBufferQueue fb_queue(FRAME_SIZE*2, 10);

int frame_counter = 0;

uint8_t tmp_buffer[FRAME_SIZE*2];

void cb(uvc_frame_t *frame, void *ptr) {
  uint8_t* data = static_cast<uint8_t*>(frame->data);
  uint8_t* left_data = tmp_buffer;
  uint8_t* right_data = tmp_buffer + FRAME_WIDTH;

  for (int i = 0; i < FRAME_HEIGHT; ++i) {
    for (int j = 0; j < FRAME_WIDTH; ++j) {
      *(left_data++) = *(data++);
      *(right_data++) = *(data++);
    }
    left_data += FRAME_WIDTH;
    right_data += FRAME_WIDTH;
  }

  frame_counter++;

  if (frame_counter != frame->sequence) {
    cout << "Missed frame(s): "
         << frame_counter << " " << frame->sequence << endl;
    frame_counter = frame->sequence;
  }

  fb_queue.addFrame(tmp_buffer);
}

void uncombine(const uint8_t* data, uint8_t* left_data, uint8_t* right_data) {
  for (int i=0; i < FRAME_HEIGHT; ++i) {
    for (int j=0; j < FRAME_WIDTH; ++j) {
      *(left_data++) = *data;
      *(right_data++) = *(data + FRAME_WIDTH);
      data++;
    }
    data += FRAME_WIDTH;
  }
}

void fail(const char* msg) {
  cerr << msg << endl;
  exit(1);
}

int main(int argc, char **argv) {
  string snapshots_dir, output_file;

  po::options_description desc("Command line options");
  desc.add_options()
      ("output-dir",
       po::value<string>(&snapshots_dir)->required(),
       "where to store snapshots");

  desc.add_options()
      ("output-file",
       po::value<string>(&output_file),
       "where to store outpit video file");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

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
          FILE* ffmpeg = 0;

          if (!output_file.empty()) {
            std::string cmd = string("ffmpeg ") +
                "-f rawvideo " +
                "-pix_fmt gray " +
                "-s 1280x480 " +
                "-r 30 " +
                "-i - " +
                "-r 30 " +
                "-c:v libx264 " +
                "-preset ultrafast " +
                "-qp 0 " +
                "-an " +
                "-f avi " +
                "-y " +
                output_file;

            cout << "Running ffmpeg: " << cmd << endl;

            ffmpeg = popen(cmd.c_str(), "w");
            if (ffmpeg == 0) {
              fail("Failed to open ffmpeg");
            }
          }

          cv::namedWindow("preview");

          uint8_t buffer[FRAME_SIZE*2];

          cv::Mat combined(FRAME_HEIGHT, FRAME_WIDTH*2, CV_8UC1, buffer);
          cv::Mat left(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);
          cv::Mat right(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);

          uvc_set_ae_mode(devh, 1); /* e.g., turn on auto exposure */

          int current_snapshot_index = 0;
          bool done = false;
          while(!done) {
            fb_queue.nextFrame(buffer);

            cv::imshow("preview", combined);

            if (ffmpeg != 0) {
              fwrite(buffer, 1, FRAME_SIZE*2, ffmpeg);
            }

            int key = cv::waitKey(1);
            switch (key) {
              case 27:
                done = true;
                break;
              case 32:
                printf("Snapshot %d!\n", ++current_snapshot_index);

                uncombine(buffer, left.ptr(), right.ptr());
                cv::imwrite(str(format("%s/left_%d.png") % snapshots_dir % current_snapshot_index), left);
                cv::imwrite(str(format("%s/right_%d.png") % snapshots_dir % current_snapshot_index), right);

                break;
              case -1:
                break;
              default:
                printf("Unknown code %d\n", key);
                break;
            }
          }

          pclose(ffmpeg);

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
