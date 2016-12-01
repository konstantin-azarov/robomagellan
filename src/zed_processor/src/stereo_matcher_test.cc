#include <chrono>
#include <iostream>
#include <random>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>


#define BACKWARD_HAS_DW 1
#include "backward.hpp"

#include "stereo_matcher.hpp"
#include "math3d.hpp"

backward::SignalHandling sh;

cv::Mat_<uchar> randomDescriptors(int n) {
  cv::Mat_<uchar> res(n, 64);

  std::default_random_engine rand(0);
  std::uniform_int_distribution<> dist(0, 0xFFFF);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 64; ++j) {
      res[i][j] = dist(rand);
    }
  }

  return res;
}

bool operator == (const ushort2& a, const ushort2& b) {
  return a.x == b.x && a.y == b.y;
}

PinnedVector<ushort2> randomPairs(int n_descs, int n_pairs) {
  PinnedVector<ushort2> res;
  std::default_random_engine rand;
  std::uniform_int_distribution<> dist(0, (n_descs*(n_descs - 1)) - 1);

  for (int t=0; t < n_pairs; ++t) {
    int r = dist(rand);
    int i = r / (n_descs - 1);
    int j = r % (n_descs - 1);
    if (j >= i) j++;

    res.push_back(make_ushort2(i, j));
  }

  std::sort(
      std::begin(res), std::end(res), 
      [](const ushort2& a, const ushort2& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
      });

  res.erase(std::unique(std::begin(res), std::end(res)), std::end(res));

  return res;
}

std::vector<cv::Vec2s> computeMatches(
    const cv::Mat_<uchar>& d1,
    const cv::Mat_<uchar>& d2,
    const PinnedVector<ushort2>& pairs,
    float threshold_ratio) {
  int n1 = d1.rows, n2 = d2.rows;

  const int kMax = 500000000;

  cv::Mat_<int> scores(n1, n2, kMax);
  
  for (auto p : pairs) {
    int s = descriptorDist(d1.row(p.x), d2.row(p.y));
    scores(p.x, p.y) = s;
  }

  /* std::cout << "Scores: " << scores << std::endl; */  

  std::vector<int> m1(n1, -1), m2(n2, -1);

  auto fill_matches = [&](std::vector<int>& m, cv::Mat_<int> scores) {
    ASSERT_EQ(m.size(), scores.rows);
    for (int i=0; i < scores.rows; ++i) {
      std::vector<cv::Vec2i> row(scores.cols);
      for (int j = 0; j < scores.cols; ++j) {
        row[j] = { scores(i, j), j };
      }

      std::sort(std::begin(row), std::end(row), [](cv::Vec2i a, cv::Vec2i b) {
        return a[0] < b[0];
      });

      /* std::cout << "Row(" << i << "): " << cv::Mat_<cv::Vec2i>(row) << std::endl; */

      if (row[0][0] != kMax && 
          row[1][0]*threshold_ratio > row[0][0]) {
        m[i] = row[0][1];
      }
    }
  };

  fill_matches(m1, scores);
  fill_matches(m2, scores.t());

  /* std::cout << "m1: " << cv::Mat_<int>(m1) << std::endl; */
  /* std::cout << "m2: " << cv::Mat_<int>(m2) << std::endl; */

  std::vector<cv::Vec2s> res;
  for (int i = 0; i < n1; ++i) {
    if (m1[i] != -1 && m2[m1[i]] == i) {
      res.push_back(cv::Vec2s(i, m1[i]));
    }
  }

  return res;
}
  
TEST(StereoMatcherTest, random) {
  const int kMaxDescs = 10000;
  const int kMaxPairs = 100000;
  
  Matcher matcher(kMaxDescs, kMaxPairs);

  std::tuple<int, int, int> tests[] = {
    std::make_tuple(10, 10, 1),
    std::make_tuple(10, 45, 1),
    std::make_tuple(1000, 1000, 5),
    std::make_tuple(1000, 50000, 10),
    std::make_tuple(10000, 100000, 1)
  };

  std::vector<cv::Vec2s> matches;

  auto stream = cv::cuda::Stream::Null();

  for (auto test : tests) {
    int ndescs = std::get<0>(test);
    int npairs = std::get<1>(test);
    int niters = std::get<2>(test);

    for (int t = 0; t < niters; ++t) {
      cv::Mat_<uchar> d1 = randomDescriptors(ndescs);
      cv::Mat_<uchar> d2 = randomDescriptors(ndescs);
      PinnedVector<ushort2> pairs = randomPairs(ndescs, npairs);
      CudaDeviceVector<ushort2> pairs_gpu(pairs.size());
      pairs_gpu.upload(pairs);

      matcher.computeScores(
          cv::cudev::GpuMat_<uint8_t>(d1), 
          cv::cudev::GpuMat_<uint8_t>(d2),
          pairs_gpu,
          pairs.size(),
          stream);
      stream.waitForCompletion();
      matcher.gatherMatches(ndescs, ndescs, pairs, 0.8, matches);

      auto golden = computeMatches(d1, d2, pairs, 0.8);

      ASSERT_EQ(golden, matches) << ndescs << ", " << npairs << ", " << t;
    }
  }
}

class MatcherBenchmarkFixture : public benchmark::Fixture {
  public:
    const int kDescs = 10000;
    const int kPairs = 100000;

    MatcherBenchmarkFixture() : pairs_gpu_(kPairs), matcher_(kDescs, kPairs) {
      matches_.reserve(kDescs);
    }

    virtual void SetUp(const benchmark::State&) {
      d1_.upload(randomDescriptors(kDescs));
      d2_.upload(randomDescriptors(kDescs));
      pairs_cpu_ = randomPairs(kDescs, kPairs);
      pairs_gpu_.upload(pairs_cpu_);
      match();
    }

    void match() {
      auto stream = cv::cuda::Stream::Null();
      matcher_.computeScores(d1_, d2_, pairs_gpu_, pairs_cpu_.size(), stream);
      stream.waitForCompletion();
      matcher_.gatherMatches(d1_.rows, d2_.rows, pairs_cpu_, 0.8, matches_);
    }

  protected:
    Matcher matcher_;
    cv::cudev::GpuMat_<uint8_t> d1_, d2_;
    PinnedVector<ushort2> pairs_cpu_;
    CudaDeviceVector<ushort2> pairs_gpu_;
    std::vector<cv::Vec2s> matches_;
};

BENCHMARK_F(MatcherBenchmarkFixture, Full)(benchmark::State& st) {
  while (st.KeepRunning()) {
    match();
  }
}

BENCHMARK_F(MatcherBenchmarkFixture, Gpu)(benchmark::State& st) {
  while (st.KeepRunning()) {
    auto stream = cv::cuda::Stream::Null();
    matcher_.computeScores(d1_, d2_, pairs_gpu_, pairs_cpu_.size(), stream);
    stream.waitForCompletion();
  }
}

BENCHMARK_F(MatcherBenchmarkFixture, Cpu)(benchmark::State& st) {
  while (st.KeepRunning()) {
    matcher_.gatherMatches(d1_.rows, d2_.rows, pairs_cpu_, 0.8, matches_);
  }
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "benchmark") {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
  } else {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }
}

