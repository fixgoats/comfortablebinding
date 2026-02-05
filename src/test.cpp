#include "Eigen/Dense"
#include <iostream>

int main(int argc, char* argv[]) {
  Eigen::Matrix3d A{1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::cout << "In memory:\n";
  for (int i = 0; i < A.size(); ++i) {
    std::cout << *(A.data() + i) << " ";
  }
  std::cout << std::endl;
  return 0;
}
