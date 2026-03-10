#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "highfive/highfive.hpp"
#include "vkcore.hpp"
#include <iostream>

using Eigen::Vector2cd, Eigen::Vector2cf;

int main(int argc, char* argv[]) {
  HighFive::File file("test.h5", HighFive::File::Truncate);
  std::vector<f64> data{1, 2, 3, 4, 5, 6, 7, 8};
  auto set = file.createDataSet<f64>("bleh", HighFive::DataSpace({2, 2, 2}));
  set.write_raw(data.data());
  return 0;
}
