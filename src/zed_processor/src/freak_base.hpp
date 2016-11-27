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

    static const int kLayers = 8;
    static const int kPointsInLayer = 6;
    // Innermost layer has one point only
    static const int kPoints = (kLayers-1)*kPointsInLayer + 1;
    static const int kOrientations = 256;
    static const int kOrientationPairs = 45;
    static const int kPairs = 512;

    double feature_size_;

    std::vector<PatternPoint> patterns_;
    std::vector<OrientationPair> orientation_pairs_;
    std::vector<DescriptorPair> descriptor_pairs_;

};

#endif
