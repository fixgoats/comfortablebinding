#include "Eigen/Dense"
#include "typedefs.h"
#include <iostream>

int main(int argc, char* argv[]) {
  Eigen::MatrixX2d A{{1.0, 2.}, {3., 4.}, {5., 6.}};

  std::cout << "In memory:\n";
  for (int i = 0; i < A.size(); ++i) {
    std::cout << *(A.data() + i) << " ";
  }
  std::cout << "\nLooping over first index in inner loop:\n";
  for (u32 i = 0; i < 2; ++i) {
    for (u32 j = 0; j < 3; ++j) {
      std::cout << A(j, i) << " ";
    }
  }
  std::cout << "\nLooping over second index in inner loop:\n";
  for (u32 i = 0; i < 3; ++i) {
    for (u32 j = 0; j < 2; ++j) {
      std::cout << A(i, j) << " ";
    }
  }
  std::cout << std::endl;
  return 0;
}
