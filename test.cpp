#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  constexpr uint N = 100;
  std::vector<double> v(N, 0);
#pragma omp parallel for
  for (uint i = 0; i < N; i++) {
    std::vector<uint> bleh = {i};
    v[i] = bleh[0];
  }
#pragma omp barrier
  for (uint i = 0; i < N; i++) {
    std::cout << v[i] << ' ';
  }
  std::cout << std::endl;
  return 0;
}
