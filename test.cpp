#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  constexpr uint N = 100;
  std::vector<double> v(N, 0);
  std::vector<std::vector<double>> vv(N);
  for
#pragma omp parallel for
    for (uint i = 0; i < N; i++) {
      v[i] +=
    }
#pragma omp barrier
  return 0;
}
