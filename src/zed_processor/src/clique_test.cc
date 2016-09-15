#define CATCH_CONFIG_MAIN

#include <cmath>
#include <random>

#include "catch.hpp"

#include "clique.hpp"

TEST_CASE("Clique manual tests", "[Clique]") {
  Clique c;
  
  c.reset(3);
  c.addEdge(0, 1);
  c.addEdge(0, 2);

  REQUIRE(c.clique() == std::vector<int>({ 0, 1 }));

  c.reset(5);
  c.addEdge(0, 1);
  c.addEdge(0, 2);
  c.addEdge(0, 4);
  c.addEdge(1, 4);

  REQUIRE(c.clique() == std::vector<int>({ 0, 1, 4 }));
}

void testRandomClique(std::default_random_engine& generator) {
  Clique c;
  int n = 100;
  int m = std::uniform_int_distribution<int>(0, (n-1)*n/2 - 1)(generator);

  auto vertex = std::bind(std::uniform_int_distribution<int>(0, n - 1), generator);

  c.reset(n);

  std::vector<std::vector<int>> g(n, std::vector<int>(n, 0));

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
        REQUIRE(g[i][j]);
      }
    }
  }
}

TEST_CASE("Clique random tests", "[Clique]") {
  std::default_random_engine generator;

  for (int t = 0; t < 100; ++t) {
    testRandomClique(generator);
  }
}
