#include "Eigen/Dense"
#include "geometry.h"
#include "highfive/highfive.hpp"
#include "spdlog/spdlog.h"
#include <cxxopts.hpp>
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

  if (result["v"].count()) {
    spdlog::set_level(spdlog::level::trace);
  }

  // auto lamb = [](f64 x, f64 y, f64 z) { return x * (y + z); };

  // std::vector<f64> a{1, 2, 3, 4, 4, 5};
  // std::vector<f64> b{1, 2, 3, 4, 4, 5};
  // std::vector<f64> c{1, 2, 3, 4, 4, 5};
  // std::vector<f64> d{0, 0, 0, 0, 0, 0};

  // apply(lamb, 6, d.data(), a.data(), b.data(), c.data());
  // for (u64 i = 0; i < 6; ++i) {
  //   std::cout << d[i] << ' ';
  // }
  // std::cout << std::endl;

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis{1, 2};

  const u64 N = 100000;
  std::vector<f64> k(N);
  for (u64 i = 0; i < N; ++i) {
    k[i] = -dis(gen);
  }

  std::vector<f64> x(N);
  for (u64 i = 0; i < N; ++i) {
    x[i] = dis(gen);
  }

  std::vector<f64> v(N, 0);

  std::vector<f64> kv1(N);
  std::vector<f64> kv2(N);
  std::vector<f64> kv3(N);
  std::vector<f64> kv4(N);
  std::vector<f64> kx1(N);
  std::vector<f64> kx2(N);
  std::vector<f64> kx3(N);
  std::vector<f64> kx4(N);
  std::vector<f64> tmpx(N);
  std::vector<f64> tmpv(N);
  const f64 dt = 0.01;

  const auto lamb = [dt](f64 y, f64 a, f64 b, f64 c, f64 d) {
    return y + (dt / 6.) * (a + (2 * b) + (2 * c) + d);
  };

  for (u64 i = 0; i < 1000; ++i) {
    mul(kv1.data(), N, k.data(), x.data());
    memcpy(kx1.data(), v.data(), v.size() * sizeof(f64));
    mul_add(tmpv.data(), v.data(), kv1.data(), 0.5 * dt, N);
    mul_add(tmpx.data(), x.data(), kx1.data(), 0.5 * dt, N);
    mul(kv2.data(), N, k.data(), tmpx.data());
    memcpy(kx2.data(), tmpv.data(), N * sizeof(f64));
    mul_add(tmpv.data(), v.data(), kv2.data(), 0.5 * dt, N);
    mul_add(tmpx.data(), x.data(), kx2.data(), 0.5 * dt, N);
    mul(kv3.data(), N, k.data(), tmpx.data());
    memcpy(kx3.data(), tmpv.data(), N * sizeof(f64));
    mul_add(tmpv.data(), v.data(), kv3.data(), dt, N);
    mul_add(tmpx.data(), x.data(), kx3.data(), dt, N);
    mul(kv4.data(), N, k.data(), tmpx.data());
    memcpy(kx4.data(), tmpv.data(), N * sizeof(f64));
    apply(lamb, N, v.data(), v.data(), kv1.data(), kv2.data(), kv3.data(),
          kv4.data());
    apply(lamb, N, x.data(), x.data(), kx1.data(), kx2.data(), kx3.data(),
          kx4.data());
  }

  HighFive::File file("test.h5", HighFive::File::Truncate);
  file.createDataSet("springs", x);

  return 0;
}
