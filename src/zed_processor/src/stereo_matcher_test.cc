#include <random>

#include <opencv2/core.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "stereo_matcher.hpp"
#include "math3d.hpp"

backward::SignalHandling sh;

cv::Mat_<uchar> randomDescriptors(int n) {
  cv::Mat_<uchar> res(n, 64);

  std::default_random_engine rand;
  std::uniform_int_distribution<> dist(0, 0xFFFF);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 64; ++j) {
      res[i][j] = dist(rand);
    }
  }

  return res;
}

std::vector<cv::Vec2s> randomPairs(int n_descs, int n_pairs) {
  std::vector<cv::Vec2s> res;
  std::default_random_engine rand;
  std::uniform_int_distribution<> dist(0, (n_descs*(n_descs - 1)) - 1);

  for (int t=0; t < n_pairs; ++t) {
    int r = dist(rand);
    int i = r / (n_descs - 1);
    int j = r % (n_descs - 1);
    if (j >= i) j++;

    res.push_back(cv::Vec2s(i, j));
  }

  return res;
}

void naiveMatch(
    const cv::Mat_<uchar>& d1,
    const cv::Mat_<uchar>& d2,
    const std::vector<cv::Vec2s>& pairs,
    std::vector<cv::Vec4s>& m1,
    std::vector<cv::Vec4s>& m2,
    std::vector<cv::Vec2s>& m) {
  m1.resize(d1.rows);
  std::fill(m1.begin(), m1.end(), cv::Vec4s(0xffff, 0xffff, 0xffff, 0));
  m2.resize(d2.rows);
  std::fill(m2.begin(), m2.end(), cv::Vec4s(0xffff, 0xffff, 0xffff, 0));

  for (const auto& p : pairs) {
    int score = descriptorDist(d1.row(p[0]), d2.row(p[1]));
    {
      cv::Vec4s m = m1[p[0]];
      if (score < m[0]) {
        m1[p[0]] = cv::Vec4s(score, m[0], p[1], 0);
      } else if (score < m[1]) {
        m1[p[0]] = cv::Vec4s(m[0], score, m[2], 0);
      }
    }
    {
      cv::Vec4s m = m2[p[1]];
      if (score < m[0]) {
        m2[p[1]] = cv::Vec4s(score, m[0], p[0], 0);
      } else if (score < m[1]) {
        m2[p[1]] = cv::Vec4s(m[0], score, m[2], 0);
      }
    }
  }

  /* for (int t = 0; t < 2; ++t) { */
  /*   for (auto& m : matches[t]) { */
  /*     if (m.x != 0xFFFF) { */
  /*       if (m.x / static_cast<double>(m.y) >= 0.8) { */
  /*         m.z = 0xFFFF; */
  /*       } */
  /*     } */
  /*   } */
  /* } */
}

int main(int argc, char** argv) {
  const int kNDescs = 5000;
  const int kNPairs = 100000;

  cv::Mat_<uchar> d1 = randomDescriptors(kNDescs);
  cv::Mat_<uchar> d2 = randomDescriptors(kNDescs);
  std::vector<cv::Vec2s> pairs = randomPairs(kNDescs, kNPairs);

  std::vector<cv::Vec4s> m1, m2;
  std::vector<cv::Vec2s> m;

  naiveMatch(d1, d2, pairs, m1, m2, m);

  cv::cudev::GpuMat_<uchar> d1_gpu(d1);
  cv::cudev::GpuMat_<uchar> d2_gpu(d2);
  cv::cudev::GpuMat_<cv::Vec2s> pairs_gpu(pairs);
  cv::cudev::GpuMat_<cv::Vec4s> m1_gpu(1, d1.rows);
  cv::cudev::GpuMat_<cv::Vec4s> m2_gpu(1, d2.rows);
  cv::cudev::GpuMat_<cv::Vec2s> m_gpu(1, 10000);

  stereo_matcher::match(d1_gpu, d2_gpu, pairs_gpu, 0.8, m1_gpu, m2_gpu, m_gpu);

  cv::Mat_<cv::Vec4s> m1_cpu, m2_cpu;
  m1_gpu.download(m1_cpu);
  m2_gpu.download(m2_cpu);

  return 0;
}
