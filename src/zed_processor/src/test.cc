#include <algorithm> 
#include <iostream>
#include <iomanip>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
  cv::FileStorage fs("/tmp/robo-debug/cross-matches-far.yml", cv::FileStorage::READ);

  std::vector<std::pair<cv::Point3d, cv::Point3d> > matches(fs["matches"].size());

  int i = 0;
  for (const auto& match : fs["matches"]) {
    match["p1"] >> matches[i].first;
    match["p2"] >> matches[i].second;
    ++i;
  }


  std::vector<int> clique_far, clique_near;
  
  cv::FileStorage fs2("/tmp/robo-debug/clique-far.yml", cv::FileStorage::READ);
  fs2["clique"] >> clique_far;
  
  cv::FileStorage fs3("/tmp/robo-debug/clique-near.yml", cv::FileStorage::READ);
  fs3["clique"] >> clique_near;


  std::cout << "Clique near: " << std::endl;
  std::cout << std::setprecision(4);
  for (int i : clique_near) {
    for (int j : clique_near) {
      const auto& m1 = matches[i];
      const auto& m2 = matches[j];

      double d1 = cv::norm(m1.first - m2.first);
      double d2 = cv::norm(m1.second - m2.second);

      std::cout << std::setw(5) << std::abs(d1 - d2) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Clique far: " << std::endl;
  for (int i : clique_far) {
    for (int j : clique_far) {
      const auto& m1 = matches[i];
      const auto& m2 = matches[j];

      double d1 = cv::norm(m1.first - m2.first);
      double d2 = cv::norm(m1.second - m2.second);

      std::cout << std::setw(6) << std::abs(d1 - d2) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Clique near far: " << std::endl;
  for (int i : clique_near) {
    for (int j : clique_far) {
      const auto& m1 = matches[i];
      const auto& m2 = matches[j];

      double d1 = cv::norm(m1.first - m2.first);
      double d2 = cv::norm(m1.second - m2.second);

      std::cout << std::setw(6) << std::abs(d1 - d2) << " ";
    }
    std::cout << std::endl;
  }
}
