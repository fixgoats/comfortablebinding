#include "Eigen/Dense"
#include "geometry.h"
#include "highfive/highfive.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cxxopts.hpp>
#include <execution>
#include <iostream>
#include <random>
#include <vector>

template <class T, class... Arrs>
void add(T* __restrict__ c, u64 n, Arrs... a) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = (0 + ... + a[i]);
  }
}

template <class T, class... Arrs>
void sub(T* __restrict__ c, u64 n, Arrs... a) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = (... - a[i]);
  }
}

template <class T, class... Arrs>
void mul(T* __restrict__ c, u64 n, Arrs... a) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = (1 * ... * a[i]);
  }
}

void mul_add(f64* __restrict__ c, const f64* a, const f64* b, const f64* d,
             u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = a[i] + d[i] * b[i];
  }
}

void mul_sub(f64* __restrict__ c, const f64* a, const f64* b, const f64* d,
             u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = a[i] - d[i] * b[i];
  }
}

void mul_add(f64* __restrict__ c, const f64* a, const f64* b, f64 d, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = a[i] + d * b[i];
  }
}

void div(f64* __restrict__ c, const f64* a, const f64* b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = a[i] / b[i];
  }
}

void add(f64* __restrict__ c, const f64* a, f64 b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = a[i] + b;
  }
}

void sub(f64* __restrict__ c, const f64* a, f64 b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = a[i] - b;
  }
}

void mul(f64* __restrict__ c, const f64* a, f64 b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = a[i] * b;
  }
}

void div(f64* __restrict__ c, const f64* a, f64 b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    c[i] = a[i] / b;
  }
}

void inplace_add(f64* __restrict__ a, const f64* b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] += b[i];
  }
}

void inplace_add(f64* __restrict__ a, f64 b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] += b;
  }
}

void inplace_mul_add(f64* __restrict__ a, const f64* b, const f64* c, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] += b[i] * c[i];
  }
}

void inplace_mul_add(f64* __restrict__ a, const f64* b, f64 c, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] += b[i] * c;
  }
}

void inplace_sub(f64* __restrict__ a, const f64* b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] -= b[i];
  }
}

void inplace_mul_sub(f64* __restrict__ a, const f64* b, const f64* c, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] -= b[i] * c[i];
  }
}

void inplace_sub(f64* __restrict__ a, f64 b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] -= b;
  }
}

void inplace_mul(f64* __restrict__ a, const f64* b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] *= b[i];
  }
}

void inplace_mul(f64* __restrict__ a, f64 b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] *= b;
  }
}

void inplace_div(f64* __restrict__ a, const f64* b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] /= b[i];
  }
}

void inplace_div(f64* __restrict__ a, f64 b, u64 n) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    a[i] /= b;
  }
}

template <class Func, class... Args>
void apply(Func f, u64 n, f64* __restrict__ dest, Args... a) {
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    dest[i] = f(a[i]...);
  }
}

void rk4step(f64* y, f64* k1, f64* k2, f64* k3, f64* k4, u64 n) {}

// f64 hooke()

int main(int argc, char* argv[]) {
  cxxopts::Options options("test program", "bleh");
  options.add_options()("v,verbose", "Verbose output", cxxopts::value<bool>());
  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<int> v = {5, 4, 3, 2, 1};
  std::sort(std::execution::seq, v.begin(), v.end());
  return 0;
}
