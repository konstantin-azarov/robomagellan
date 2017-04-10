#ifndef __CLIQUE_H__
#define __CLIQUE_H__

#include <stdint.h>
#include <vector>

class Clique {
  public:
    Clique() : n_(0) {}

    void reset(int n);
    // Colors must be set before edges are added
    void setColor(int i, int c);
    void addEdge(int i, int j);

    const std::vector<int>& compute();

    const std::vector<int>& clique() const { return clique_; }

  private:
    int edge_(int i, int j) { return i*n_ + j; }

  private:
    std::vector<uint8_t> graph_;
    std::vector<int> degrees_;
    std::vector<int> clique_;
    std::vector<int> candidates_[2];
    std::vector<int> colors_;
    int n_;
};

#endif
