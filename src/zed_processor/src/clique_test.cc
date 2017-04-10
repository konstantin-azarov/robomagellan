#define CATCH_CONFIG_MAIN

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>

#include <gtest/gtest.h>

#include "clique.hpp"

TEST(CliqueTest, clique) {
  Clique c;
  
  c.reset(3);
  for (int i=0; i < 3; ++i) {
    c.setColor(i, i);
  }
  c.addEdge(0, 1);
  c.addEdge(0, 2);

  ASSERT_EQ(std::vector<int>({ 0, 1 }), c.compute());

  c.reset(5);
  for (int i=0; i < 5; ++i) {
    c.setColor(i, i);
  }
  c.addEdge(0, 1);
  c.addEdge(0, 2);
  c.addEdge(0, 4);
  c.addEdge(1, 4);

  ASSERT_EQ(std::vector<int>({ 0, 1, 4 }), c.compute());
}

TEST(CliqueTest, cliqueColored) {
  Clique c;
  
  c.reset(6);
  c.setColor(0, 0);
  c.setColor(1, 0);
  c.setColor(2, 0);
  c.setColor(3, 1);
  c.setColor(4, 1);
  c.setColor(5, 6);
  c.addEdge(0, 1);
  c.addEdge(0, 2);
  c.addEdge(0, 5);
  c.addEdge(1, 2);
  c.addEdge(2, 4);
  c.addEdge(2, 5);
  c.addEdge(3, 4);
  c.addEdge(4, 5);
  
  auto res = c.compute();
  std::sort(std::begin(res), std::end(res));
  ASSERT_EQ(std::vector<int>({2, 4, 5}), res);
}

void testRandomClique(std::default_random_engine& generator) {
  Clique c;
  int n = 100;
  int m = std::uniform_int_distribution<int>(0, (n-1)*n/2 - 1)(generator);

  auto vertex = std::bind(std::uniform_int_distribution<int>(0, n - 1), generator);

  c.reset(n);

  std::vector<std::vector<int>> g(n, std::vector<int>(n, 0));

  for (int i=0; i < n; ++i) {
    c.setColor(i, i);
  }

  for (int t=0; t < m; ++t) {
    int i = vertex();
    int j = vertex();
    c.addEdge(i, j);
    g[i][j] = g[j][i] = 1;
  }

  const auto& res = c.clique();
  for (int i : res) {
    for (int j : res) {
      if (i != j) {
        ASSERT_TRUE(g[i][j]);
      }
    }
  }
}

TEST(CliqueTest, randomClique) {
  std::default_random_engine generator;

  for (int t = 0; t < 100; ++t) {
    testRandomClique(generator);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

