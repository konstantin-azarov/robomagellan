#include <assert.h>

#include "clique.hpp"

void Clique::reset(int n) {
  n_ = n;
  graph_.resize(n*n);
  std::fill(graph_.begin(), graph_.end(), 0);
  degrees_.resize(n);
  std::fill(degrees_.begin(), degrees_.end(), 0);
  colors_.resize(n);
}

void Clique::setColor(int i, int c) {
  colors_[i] = c;
}

void Clique::addEdge(int i, int j) {
  assert(i < n_);
  assert(j < n_);

  if (!graph_[edge_(i, j)]) { 
    graph_[edge_(i, j)] = 1;
    graph_[edge_(j, i)] = 1;
    if (colors_[i] != colors_[j]) {
      degrees_[i]++;
      degrees_[j]++;
    }
  }
}

#include <iostream>

const std::vector<int>& Clique::compute() {
  assert(n_ > 0);

  clique_.resize(0);

  int best = 0;
  for (int i = 1; i < n_; ++i) {
    if (degrees_[i] > degrees_[best]) {
      best = i;
    }
  }

  candidates_[0].resize(0);
  for (int i=0; i < n_; ++i) {
    if (i != best) {
      candidates_[0].push_back(i);
    }
  }

  int t = 0;
  while (best >= 0) {
    clique_.push_back(best);
    int c = colors_[best];
  
    candidates_[1-t].resize(0);
    for (int i=0; i < (int)candidates_[t].size(); ++i) {
      if (c != colors_[candidates_[t][i]] && 
          graph_[edge_(best, candidates_[t][i])]) {
        candidates_[1-t].push_back(candidates_[t][i]);
      }
    }

    best = -1;
    for (int i=0; i < (int)candidates_[1-t].size(); ++i) {
      if (best == -1 || degrees_[candidates_[1-t][i]] > degrees_[best]) {
       best = candidates_[1-t][i];
      }
    }

    t = 1-t;
  }

  return clique_;
}
