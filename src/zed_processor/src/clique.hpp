#ifndef __CLIQUE_H__
#define __CLIQUE_H__

#include <stdint.h>
#include <vector>

class Clique {
  public:
    Clique() : n_(0) {}

    void reset(int n);
    void addEdge(int i, int j);
    const std::vector<int>& clique();

  private:
    int edge_(int i, int j) { return i*n_ + j; }

  private:
    std::vector<uint8_t> graph_;
    std::vector<int> degrees_;
    std::vector<int> clique_;
    std::vector<int> candidates_[2];
    int n_;
    bool computed_;
};

#endif
