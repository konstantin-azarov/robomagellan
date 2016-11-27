#include <opencv2/core.hpp>

#include "freak_base.hpp"

const std::vector<int> kSelectedPairs = {
     404,431,818,511,181,52,311,874,774,543,719,230,417,205,11,
     560,149,265,39,306,165,857,250,8,61,15,55,717,44,412,
     592,134,761,695,660,782,625,487,549,516,271,665,762,392,178,
     796,773,31,672,845,548,794,677,654,241,831,225,238,849,83,
     691,484,826,707,122,517,583,731,328,339,571,475,394,472,580,
     381,137,93,380,327,619,729,808,218,213,459,141,806,341,95,
     382,568,124,750,193,749,706,843,79,199,317,329,768,198,100,
     466,613,78,562,783,689,136,838,94,142,164,679,219,419,366,
     418,423,77,89,523,259,683,312,555,20,470,684,123,458,453,833,
     72,113,253,108,313,25,153,648,411,607,618,128,305,232,301,84,
     56,264,371,46,407,360,38,99,176,710,114,578,66,372,653,
     129,359,424,159,821,10,323,393,5,340,891,9,790,47,0,175,346,
     236,26,172,147,574,561,32,294,429,724,755,398,787,288,299,
     769,565,767,722,757,224,465,723,498,467,235,127,802,446,233,
     544,482,800,318,16,532,801,441,554,173,60,530,713,469,30,
     212,630,899,170,266,799,88,49,512,399,23,500,107,524,90,
     194,143,135,192,206,345,148,71,119,101,563,870,158,254,214,
     276,464,332,725,188,385,24,476,40,231,620,171,258,67,109,
     844,244,187,388,701,690,50,7,850,479,48,522,22,154,12,659,
     736,655,577,737,830,811,174,21,237,335,353,234,53,270,62,
     182,45,177,245,812,673,355,556,612,166,204,54,248,365,226,
     242,452,700,685,573,14,842,481,468,781,564,416,179,405,35,
     819,608,624,367,98,643,448,2,460,676,440,240,130,146,184,
     185,430,65,807,377,82,121,708,239,310,138,596,730,575,477,
     851,797,247,27,85,586,307,779,326,494,856,324,827,96,748,
     13,397,125,688,702,92,293,716,277,140,112,4,80,855,839,1,
     413,347,584,493,289,696,19,751,379,76,73,115,6,590,183,734,
     197,483,217,344,330,400,186,243,587,220,780,200,793,246,824,
     41,735,579,81,703,322,760,720,139,480,490,91,814,813,163,
     152,488,763,263,425,410,576,120,319,668,150,160,302,491,515,
     260,145,428,97,251,395,272,252,18,106,358,854,485,144,550,
     131,133,378,68,102,104,58,361,275,209,697,582,338,742,589,
     325,408,229,28,304,191,189,110,126,486,211,547,533,70,215,
     670,249,36,581,389,605,331,518,442,822
};

FreakBase::FreakBase(double feature_size) 
    : feature_size_(feature_size) {
  buildPatterns_();
}

int FreakBase::borderWidth() const { 
  return ceil(feature_size_) + 1;
}

void FreakBase::buildPatterns_() {
  patterns_.resize(kOrientations * kPoints);

  const int layers = 8;
  const int points_in_layer = 6;

  // pattern definition, radius normalized to 1.0 
  // (outer point position+sigma=1.0)
  const double bigR = 2.0/3.0; 
  const double smallR = 2.0/24.0;
  const double unitSpace = (bigR-smallR)/21.0; 
  // radii of the concentric cirles (from outer to inner)
  const double radius[8] = {
    bigR, 
    bigR-6*unitSpace, 
    bigR-11*unitSpace, 
    bigR-15*unitSpace, 
    bigR-18*unitSpace, 
    bigR-20*unitSpace, 
    smallR, 
    0.0};

  // sigma of pattern points (each group of 6 points on a concentric cirle has 
  // the same sigma)
  const double sigma[8] = {
    radius[0]/2.0, 
    radius[1]/2.0, 
    radius[2]/2.0,
    radius[3]/2.0, 
    radius[4]/2.0, 
    radius[5]/2.0,
    radius[6]/2.0, 
    radius[6]/2.0
  };

  // fill the points table
  for(int orientation = 0; orientation < kOrientations; ++orientation) {
    double theta = double(orientation)* 2*CV_PI/double(kOrientations);

    PatternPoint* points = &patterns_[orientation*kPoints];
    int idx = 0;
    for( size_t l = 0; l < layers; ++l )
    {
      double r = radius[l];
      int n = r > 0 ? points_in_layer : 1;
      double beta = CV_PI/n * (l%2); 
      double s = sigma[l];

      for( int k = 0 ; k < n; ++k )
      {
          // orientation offset so that groups of points on each circles are 
          // staggered
          double alpha = k*2.0*CV_PI/n + beta + theta;

          // add the point to the look-up table
          PatternPoint& point = points[idx++];
          point.x = static_cast<float>(r * cos(alpha) * feature_size_);
          point.y = static_cast<float>(r * sin(alpha) * feature_size_);
          point.sigma = static_cast<float>(s * feature_size_);
      }
    }
  }

  // fill the pairs table
  std::vector<DescriptorPair> all_pairs;
  for (int i = 1; i < kPoints; ++i) {
    for (int j = 0; j < i; ++j) {
      all_pairs.push_back(DescriptorPair { i, j });
    }
  }

  descriptor_pairs_.resize(kSelectedPairs.size());
  for (int i=0; i < kSelectedPairs.size(); ++ i) {
    descriptor_pairs_[i] = all_pairs[kSelectedPairs[i]];
  }
  

  // fill the orientation pairs table
  for (int l = 0; l < kLayers - 1; ++l) {
    // Symmetrical points
    for (int i=0; i < kPointsInLayer/2; ++i) {
      int j = (i + kPointsInLayer/2) % kPointsInLayer;
      orientation_pairs_.push_back(OrientationPair { 
          i + l*kPointsInLayer, j + l*kPointsInLayer, 0, 0 });
    }
    // Offset by 2
    if (l <= 3) {
      for (int i=0; i < kPointsInLayer; ++i) {
        int j = (i + 2) % kPointsInLayer;
        orientation_pairs_.push_back(OrientationPair { 
            i + l*kPointsInLayer, j + l*kPointsInLayer, 0, 0 });
      }
    }
  }

  if (orientation_pairs_.size() != kOrientationPairs) {
    abort();
  }

  double f = 22.0 / feature_size_;
  for (auto& o : orientation_pairs_) {
    const float dx = (patterns_[o.i].x - patterns_[o.j].x) * f;
    const float dy = (patterns_[o.i].y - patterns_[o.j].y) * f;
    const float norm_sq = (dx*dx+dy*dy);
    o.dx = static_cast<int>(dx/norm_sq*4096.0+0.5);
    o.dy = static_cast<int>(dy/norm_sq*4096.0+0.5);
  }
}

