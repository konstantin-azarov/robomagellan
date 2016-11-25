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

  std::sort(
      std::begin(res), std::end(res), 
      [](const cv::Vec2d& a, const cv::Vec2d& b) {
        return a[0] != b[0] ? a[0] < b[0] : a[1] < b[1];
      });

  res.erase(std::unique(std::begin(res), std::end(res)), std::end(res));

  return res;
}

std::vector<cv::Vec2s> computeMatches(
    const cv::Mat_<uchar>& d1,
    const cv::Mat_<uchar>& d2,
    const std::vector<cv::Vec2s>& pairs,
    float threshold_ratio) {
  int n1 = d1.rows, n2 = d2.rows;

  const int kMax = 500000000;

  cv::Mat_<int> scores(n1, n2, kMax);
  
  for (auto p : pairs) {
    int s = descriptorDist(d1.row(p[0]), d2.row(p[1]));
    scores(p[0], p[1]) = s;
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
      std::vector<cv::Vec2s> pairs = randomPairs(ndescs, npairs);

       matcher.computeScores(
          cv::cudev::GpuMat_<uint8_t>(d1), 
          cv::cudev::GpuMat_<uint8_t>(d2),
          cv::cudev::GpuMat_<cv::Vec2s>(pairs), 
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

    MatcherBenchmarkFixture() : matcher_(kDescs, kPairs) {
      matches_.reserve(kDescs);
    }

    virtual void SetUp(const benchmark::State&) {
      d1_.upload(randomDescriptors(kDescs));
      d2_.upload(randomDescriptors(kDescs));
      pairs_cpu_ =randomPairs(kDescs, kPairs);
      pairs_gpu_.upload(pairs_cpu_);
      match();
    }

    void match() {
      auto stream = cv::cuda::Stream::Null();
      matcher_.computeScores(d1_, d2_, pairs_gpu_, stream);
      stream.waitForCompletion();
      matcher_.gatherMatches(d1_.rows, d2_.rows, pairs_cpu_, 0.8, matches_);
    }

  protected:
    Matcher matcher_;
    cv::cudev::GpuMat_<uint8_t> d1_, d2_;
    std::vector<cv::Vec2s> pairs_cpu_;
    cv::cudev::GpuMat_<cv::Vec2s> pairs_gpu_;
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
    matcher_.computeScores(d1_, d2_, pairs_gpu_, stream);
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


/* template <class T> */
/* bool compare(const std::vector<T> v1, */
/*              const std::vector<T> v2) { */
/*   if (v1.size() != v2.size()) { */
/*     std::cout << "Size mismatch" << std::endl; */
/*     return false; */
/*   } */

/*   for (int i=0; i < v1.size(); ++i) { */
/*     if (v1[i] != v2[i]) { */
/*       std::cout << "Mismatch at " << i << ": " */ 
/*         << v1[i] << " != " << v2[i] << std::endl; */
/*       return false; */
/*     } */
/*   } */

/*   return true; */
/* } */

/* int main(int argc, char** argv) { */
/*   const int kNDescs = 5000; */
/*   const int kNPairs = 100000; */

/*   cv::Mat_<uchar> d1 = randomDescriptors(kNDescs); */
/*   cv::Mat_<uchar> d2 = randomDescriptors(kNDescs); */
/*   std::vector<cv::Vec2s> pairs = randomPairs(kNDescs, kNPairs); */

/*   sort(pairs.begin(), pairs.end(), [](const cv::Vec2s& a, const cv::Vec2s&b) { */
/*     return a[0] != b[0] ? a[0] < b[0] : a[1] < b[1]; */
/*   }); */

/*   std::vector<cv::Vec4w> m1, m2; */
/*   std::vector<cv::Vec2s> m; */
/*   std::vector<ushort> scores; */

/*   twoStepMatch(d1, d2, pairs, scores, m1, m2); */

/*   cv::cudev::GpuMat_<uchar> d1_gpu(d1); */
/*   cv::cudev::GpuMat_<uchar> d2_gpu(d2); */
/*   cv::cudev::GpuMat_<cv::Vec2s> pairs_gpu(pairs); */
/*   cv::cudev::GpuMat_<ushort> scores_gpu(1, pairs_gpu.cols); */

/*   stereo_matcher::scores(d1_gpu, d2_gpu, pairs_gpu, scores_gpu); */

/*   std::vector<ushort> scores_cpu; */
/*   scores_gpu.download(scores_cpu); */

/*   std::cout << "Comparing scores" << std::endl; */
/*   compare(scores, scores_cpu); */ 

/*   /1* std::cout << "Comparing m1" << std::endl; *1/ */
/*   /1* if (!compare(m1, m1_cpu)) { *1/ */
/*   /1*   return 1; *1/ */
/*   /1* } *1/ */

/*   /1* std::cout << "Comparing m2" << std::endl; *1/ */
/*   /1* if (!compare(m2, m2_cpu)) { *1/ */
/*   /1*   return 1; *1/ */
/*   /1* } *1/ */

/*   std::cout << "OK" << std::endl; */

/*   if (0) { */
/*     const int kIters = 100; */

/*     for (int t = 0; t < 20; ++t) { */
/*       twoStepMatch(d1, d2, pairs, scores, m1, m2); */
/*     } */

/*     auto t0 = std::chrono::high_resolution_clock::now(); */


/*     for (int t = 0; t < kIters; ++t) { */
/* //      twoStepMatch(d1, d2, pairs, scores, m1, m2); */
/*       foldScores(pairs, scores, m1, m2); */
/*     } */

/*     auto t1 = std::chrono::high_resolution_clock::now(); */

/*     std::cout << "CPU: " */ 
/*       << (std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / kIters) */
/*       << "us" << std::endl; */
/*   } */

/*   if (1) { */
/*     const int kIters = 500; */

/*     for (int t = 0; t < 100; ++t) { */
/*       stereo_matcher::scores(d1_gpu, d2_gpu, pairs_gpu, scores_gpu); */
/*       cudaDeviceSynchronize(); */
/*     } */

/*     auto t0 = std::chrono::high_resolution_clock::now(); */


/*     for (int t = 0; t < kIters; ++t) { */
/*       stereo_matcher::scores(d1_gpu, d2_gpu, pairs_gpu, scores_gpu); */
/*       cudaDeviceSynchronize(); */
/*     } */

/*     auto t1 = std::chrono::high_resolution_clock::now(); */

/*     std::cout << "GPU: " */ 
/*       << (std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / kIters) */
/*       << "us" << std::endl; */
/*   } */

/*   return 0; */
/* } */
