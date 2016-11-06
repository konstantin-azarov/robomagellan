#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "fast_gpu.hpp"

namespace po = boost::program_options;

backward::SignalHandling sh;

void fast_scores(const cv::Mat_<uint8_t>& img, cv::Mat_<uint8_t>& scores) {
  for (int y = 3; y < img.rows - 3; ++y) {
    for (int x = 3; x < img.cols - 3; ++x) {
      int v0 = img(y, x);
      int vs[16] = { 
        img(y - 3, x - 1),
        img(y - 3, x),
        img(y - 3, x + 1),
        img(y - 2, x + 2),
        img(y - 1, x + 3),
        img(y    , x + 3),
        img(y + 1, x + 3),
        img(y + 2, x + 2),
        img(y + 3, x + 1),
        img(y + 3, x),
        img(y + 3, x - 1),
        img(y + 2, x - 2),
        img(y + 1, x - 3),
        img(y    , x - 3),
        img(y - 1, x - 3),
        img(y - 2, x - 2)
      };

      int score = 0;
      for (int i0=0; i0 < 16; i0++) {
        int sp = std::max(0, vs[i0] - v0);
        int sn = std::max(0, v0 - vs[i0]);
        for (int i = 1; i < 9; ++i) {
          int j = (i + i0) % 16;
          sp = std::min(sp, std::max(0, vs[j] - v0));
          sn = std::min(sn, std::max(0, v0 - vs[j]));
        }
        score = std::max(score, std::max(sp, sn));
      }

      scores(y, x) = score;
    }
  }
}

void nonmax_supression(
    cv::Mat_<uint8_t>& scores, 
    std::vector<cv::KeyPoint>& kps,
    int threshold) {
  for (int y = 3; y < scores.rows - 3; ++y) {
    for (int x = 3; x < scores.cols - 3; ++x) {
      int score = scores(y, x);

      bool is_max =
          score > scores(y - 1, x - 1) &&
          score > scores(y - 1, x    ) &&
          score > scores(y - 1, x + 1) &&

          score > scores(y    , x - 1) &&
          score > scores(y    , x + 1) &&

          score > scores(y + 1, x - 1) &&
          score > scores(y + 1, x    ) &&
          score > scores(y + 1, x + 1);

      if (is_max && score > threshold) {
        kps.push_back(
            cv::KeyPoint(cv::Point2f(x, y), 0, 0, score));
      }
    }
  }
}

std::vector<cv::KeyPoint> fast_detect(const cv::Mat_<uint8_t>& img, int threshold) {
  cv::Mat_<uint8_t> scores(img.rows, img.cols, (uint8_t)0);
  fast_scores(img, scores);

  std::vector<cv::KeyPoint> res;
  nonmax_supression(scores, res, threshold);
  return res;
}

void validateScores(
    const cv::Mat_<uint8_t>& img, 
    const cv::cudev::GpuMat_<uint8_t>& gpu_img,
    int threshold) {
  cv::Mat_<uint8_t> scores(img.rows, img.cols, (uint8_t)0);

  fast_scores(img, scores);

  auto cpu_kp = fast_detect(img, threshold);

  FastGpu fast_gpu(50000, 3);
  fast_gpu.detect(gpu_img, threshold);

  cv::Mat_<uint8_t> scores1;
  fast_gpu.scores().download(scores1);

  std::vector<cv::Vec3s> gpu_kp;
  fast_gpu.keypoints().download(gpu_kp);

  sort(cpu_kp.begin(), cpu_kp.end(), [](cv::KeyPoint a, cv::KeyPoint b) {
        if (a.pt.y != b.pt.y) {
          return a.pt.y < b.pt.y;
        }
        return a.pt.x < b.pt.x;
      });

  /* for (int i=0; i < gpu_kp.size(); ++i) { */
  /*   std::cout << "!!" << gpu_kp[i][0] << ", " << gpu_kp[i][1] << std::endl; */
  /* } */

  sort(gpu_kp.begin(), gpu_kp.end(), [](cv::Vec3s a, cv::Vec3s b) {
        if (a[1] != b[1]) {
          return a[1] < b[1];
        }
        return a[0] < b[0];
      });

  int x = 101, y = 6;

  /* std::cout << "Scores: " << std::endl; */
  /* for (int i=-1; i <= 1; ++i) { */
  /*   for (int j=-1; j <= 1; ++j) { */
  /*     std::cout << (int)scores(i + y, j + x) << " "; */
  /*   } */
  /*   std::cout << std::endl; */
  /* } */


  std::cout << "Keypoints: " << cpu_kp.size() << ", " << gpu_kp.size() << std::endl;
  for (int i = 0; i < cpu_kp.size(); ++i) {
    if (gpu_kp[i][0] != cpu_kp[i].pt.x || 
        gpu_kp[i][1] != cpu_kp[i].pt.y ||
        gpu_kp[i][2] != cpu_kp[i].response) {
      std::cout << "Keypoint mismatch: " 
        << i << " " 
        << "cpu=(" << cpu_kp[i].pt.x << ", " << cpu_kp[i].pt.y << ") "
        << "gpu=(" << gpu_kp[i][0] << ", " << gpu_kp[i][1] << ")"
        << std::endl;
      return;
    }
  }
 
  /* std::cout << "CPU tile:" << std::endl; */
  /* for (int i = 0; i < 22; ++i) { */
  /*   for (int j = 0; j < 22; ++j) { */
  /*     std::cout << (int)img(i + 128, j + 1040) << " "; */
  /*   } */
  /*   std::cout << std::endl; */
  /* } */

  /* for (int i=0; i < scores1.rows; ++i) { */
  /*   for (int j=0; j < scores1.cols; ++j) { */
  /*     if (scores1(i, j) > 0) { */
  /*       std::cout << "hit " << i << " " << j << std::endl; */
  /*     } */
  /*   } */
  /* } */

  for (int i = 0; i < scores.rows; ++i) {
    for (int j = 0; j < scores.cols; ++j) {
      if (scores(i, j) != scores1(i, j) && scores(i, j) > threshold) {
        std::cout << "Mismatch @(" 
          << i << ", " << j << ")"
          << (int)scores(i, j) << " " << (int)scores1(i, j) << std::endl;
        return;
      }
    }
  }

  std::cout << "Done!" << std::endl;
}

int main(int argc, char** argv) {
  std::string img_file;
  int threshold;

  po::options_description options("FAST performance test");
  options.add_options()
      ("image",
       po::value<std::string>(&img_file)->required(),
       "path to the image fil")
      ("threshold",
       po::value<int>(&threshold)->default_value(25),
       "FAST threshold");
  
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  
  auto tmp = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
  cv::Mat_<uint8_t> img;
  cv::resize(tmp, img, cv::Size(1280, 720));

  cv::cudev::GpuMat_<uint8_t> img_gpu(img);
  
  std::vector<cv::KeyPoint> kp;

  auto fast = cv::cuda::FastFeatureDetector::create(
      threshold, true, cv::cuda::FastFeatureDetector::TYPE_9_16, 50000);
  fast->detect(img_gpu, kp);

  std::vector<cv::KeyPoint> kp1 = fast_detect(img, threshold);

  std::cout << "D: " << kp.size() << " " << kp1.size() << std::endl;
  if (kp.size() == kp1.size()) {
    for (int i=0; i < kp.size(); ++i) {
      auto k1 = kp[i];
      auto k2 = kp[i];
      if (k1.pt != k2.pt || k1.response != k2.response) {
        std::cout << "Mismatch" << std::endl;
        abort();
      }
    }
  }

  validateScores(img, img_gpu, threshold);

  /* return 0; */

  if (1) {
    const int kIters = 500;
    cv::cuda::GpuMat kp_gpu;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < kIters; ++t) {
      kp.clear();
      auto fast = cv::cuda::FastFeatureDetector::create(
          threshold, true, cv::cuda::FastFeatureDetector::TYPE_9_16, 50000);
      fast->detectAsync(img_gpu, kp_gpu);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Keypoints: " << kp_gpu.size() << std::endl;
    std::cout << "OpenCV: " 
      << (std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / kIters)
      << "ms" << std::endl;
  }

  if (1) {
    const int kIters = 1000;

    FastGpu fast_gpu(50000, 3);

    cv::cudev::GpuMat_<uint8_t> scores(img.rows, img.cols);
    scores.setTo(0);

    for (int t = 0; t < 100; ++t) {
//      kp.clear();
      fast_gpu.detect(img_gpu, threshold);
    }

    auto t0 = std::chrono::high_resolution_clock::now();


    for (int t = 0; t < kIters; ++t) {
//      kp.clear();
      fast_gpu.detect(img_gpu, threshold);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "My: " 
      << (std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / kIters)
      << "ms" << std::endl;
  }
}
