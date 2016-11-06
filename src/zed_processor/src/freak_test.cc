#include <boost/program_options.hpp>

#include <chrono>
#include <iostream>

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "freak.hpp"
#include "freak_gpu.hpp"

namespace po = boost::program_options;

backward::SignalHandling sh;

cv::Mat_<uint8_t> linear_to_opencv(cv::Mat_<uint8_t> m) {
  if (m.cols != 512/8) {
    abort();
  }

  cv::Mat_<uint8_t> res = cv::Mat_<uint8_t>::zeros(1, 512/8);

  for (int j = 0; j < 512; ++j) {
    int v = (m[0][j/8] & (1 << (j%8))) ? 1 : 0;
    int j1 = j % 128;
    int b = ( 15 - (j1 % 16)) * 8 + 7 - j1/16 + (j/128)*128;
    res[0][b/8] |= v << (7 - (b%8));
  }

  return res;
}

cv::Mat_<uint8_t> gpu_to_linear(cv::Mat_<uint8_t> m) {
  if (m.cols != 512/8) {
    abort();
  }

  cv::Mat_<uint8_t> res = cv::Mat_<uint8_t>::zeros(1, 512/8);
 
  for (int t = 0; t < 64; ++t) {
    for (int j = 0; j < 8; ++j) {
      int v = (m[0][t] & (1 << j)) ? 1 : 0;
      int b = t + j*64;
      res[0][b/8] |= v << (b%8);
      b++;
    }
  }
  return res;
}

int main(int argc, char** argv) {
  std::string img_file;
  int threshold;
  double feature_size;
  
  po::options_description options("Command line options");
  options.add_options()
      ("image",
       po::value<std::string>(&img_file)->required(),
       "path to the image fil")
      ("threshold",
       po::value<int>(&threshold)->default_value(30),
       "FAST threshold")
      ("feature-size",
       po::value<double>(&feature_size)->default_value(64.980350),
       "FREAK feature size");
  
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  int n_dev = cv::cuda::getCudaEnabledDeviceCount();
  for (int i=0; i < n_dev; ++i) {
    std::cout << "Device " << i << std::endl;
    cv::cuda::printCudaDeviceInfo(i);
  }


  std::cout << "Freak test" << std::endl;

  auto tmp = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
  cv::Mat img;
  cv::resize(tmp, img, cv::Size(1280, 720));

  std::vector<cv::KeyPoint> kp = {
    cv::KeyPoint(cv::Point2f(300, 400), 1)
  };


  cv::FAST(img, kp, threshold, true);
  
  auto kp2 = kp;

  auto opencv_freak = cv::xfeatures2d::FREAK::create(true, false);
  Freak freak(feature_size);
  FreakGpu freak_gpu(feature_size);

  cv::Mat_<uint8_t> opencv_descriptors;
  opencv_freak->compute(img, kp, opencv_descriptors);

  const cv::Mat_<uint8_t>& desc = (const cv::Mat_<uint8_t>&)freak.describe(img, kp2);

  std::vector<cv::KeyPoint> gpu_kp = kp;
  /* { */
  /*   kp[0], */
  /*   kp[1], */
  /*   kp[2], */
  /*   kp[3], */
  /*   kp[4] */
  /* }; */

//  const cv::Mat_<uint8_t>& desc = (const cv::Mat_<uint8_t>&)freak.describe(img, kp2);
  cv::Mat_<cv::Vec3s> keypoints_cpu(1, gpu_kp.size());
  for (int i = 0; i < gpu_kp.size(); ++i) {
    keypoints_cpu(0, i) = cv::Vec3s(gpu_kp[i].pt.x, gpu_kp[i].pt.y, 0);
  }

  cv::cuda::GpuMat img_gpu(img);
  cv::cuda::GpuMat keypoints_gpu(keypoints_cpu);
  freak_gpu.describe(img_gpu, keypoints_gpu); 
  auto gpu_desc = freak_gpu.descriptorsCpu();


  /* std::cout */ 
  /*   << "OpenCV: p=" << kp[0].pt */ 
  /*   << ", angle=" << kp[0].angle */ 
  /*   << ", d = " << opencv_descriptors.row(0) */
  /*   << std::endl; */

  std::cout 
    << "New: p=" << kp2[0].pt 
    << ", angle=" << kp2[0].angle 
    << ", d = " << desc.row(0)
    << std::endl;

  std::cout
    << "Gpu:"
    << " d=" << gpu_to_linear(gpu_desc.row(0))
    << std::endl;

  for (int i=0; i < gpu_kp.size(); ++i) {
    bool good = !cv::countNonZero(
        desc.row(i) != gpu_to_linear(gpu_desc.row(i)));

    if (!good) {
      std::cout << "Descriptor mismatch: " << i << std::endl;
      abort();
    }
  }

  /* std::cout */ 
  /*   << "New: p=" */ 
  /*   << kp2[0].pt */ 
  /*   << ", angle=" */ 
  /*   << kp2[0].angle */
  /*   << ", d = " << remap(desc.row(0)) */
  /*   << std::endl; */

  if (kp.size() != kp2.size()) {
    std::cout 
      << "Sizes don't match: " 
      << kp.size() << ", " << kp2.size() 
      << std::endl;
  } else {
    for (int i=0; i < kp.size(); i++) {
      const auto& k1 = kp[i];
      const auto& k2 = kp2[i];

      if (fabs(k1.angle - k2.angle) > 1E-5) {
        std::cout << "Angle mismatch for " << i << ": " 
          << k1.angle << ", " << k2.angle << std::endl;
      }

      bool good = !cv::countNonZero(
          opencv_descriptors.row(i) != linear_to_opencv(desc.row(i)));

      if (!good) {
        std::cout << "Descriptor mismatch for " << i << ": "
          << opencv_descriptors.row(i) << ", " << linear_to_opencv(desc.row(i)) 
          << std::endl;
      }
    }
  }

  std::cout << "Done matching" << std::endl;

  if (1) {
 
    const int kIters = 500;
    auto t0 = std::chrono::high_resolution_clock::now();
 
    for (int t = 0; t < kIters; ++t) {
      freak_gpu.describe(img_gpu, keypoints_gpu);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Descriptors: " << kp.size() << std::endl;
    std::cout << "My GPU: " 
      << (std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / kIters)
      << "ms" << std::endl;
  }

  if (0) {
    const int kIters = 500;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < kIters; ++t) {
      freak.describe(img, kp2);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Descriptors: " << kp.size() << std::endl;
    std::cout << "My CPU: " 
      << (std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / kIters)
      << "ms" << std::endl;
  }  
  
  if (0) {
    const int kIters = 500;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < kIters; ++t) {
      opencv_freak->compute(img, kp, opencv_descriptors);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Descriptors: " << kp.size() << std::endl;
    std::cout << "OpenCV: " 
      << (std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / kIters)
      << "ms" << std::endl;
  }


}
