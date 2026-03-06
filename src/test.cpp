#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "vkcore.hpp"
#include <iostream>

using Eigen::Vector2cd, Eigen::Vector2cf;

int main(int argc, char* argv[]) {
  Eigen::VectorXd a(10);
  a << 1, 2, 3, 4, 5, 6, 7, 8, 8, 9;
  Eigen::VectorXd b = a;
  a(3) = 5;
  std::cout << b(3) << '\n';
  return 0;
}
