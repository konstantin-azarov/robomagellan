#ifndef __FREAK_BASE__HPP__
#define __FREAK_BASE__HPP__

#include <vector>

class FreakBase {
  public:
    FreakBase(double feature_size);

    int borderWidth() const;

  protected:
    void buildPatterns_();

  protected:
    struct PatternPoint {
      float x, y, sigma;
    };

    struct OrientationPair {
      int i, j;
      int dx, dy;
    };

    struct DescriptorPair {
      int i, j;
    };

    const int kLayers = 8;
    const int kPointsInLayer = 6;
    // Innermost layer has one point only
    const int kPoints = (kLayers-1)*kPointsInLayer + 1;
    const int kOrientations = 256;
    const int kPairs = 512;

    double feature_size_;

    std::vector<PatternPoint> patterns_;
    std::vector<OrientationPair> orientation_pairs_;
    std::vector<DescriptorPair> descriptor_pairs_;

};

#endif
