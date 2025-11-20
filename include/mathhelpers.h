#pragma once
#include "Eigen/Dense"
#include "typedefs.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <vector>
#ifndef NDEBUG
#include <cassert>
#endif // !DEBUG

constexpr f32 hbar = 6.582119569e-1;
constexpr f32 muB = 5.7883818060e-2;
constexpr f32 echarge = 1e3;

using Eigen::Vector2d;
using std::bit_cast;

static constexpr f64 expCoupling(Vector2d v) { return std::exp(-v.norm()); }

template <typename T>
constexpr u32 euclid_mod(T a, u32 b) {
#ifndef NDEBUG
  assert(b != 0);
#endif
  return a % b;
}

constexpr bool fleq(auto x, auto y, double tol) {
  return std::abs(y - x) < tol;
}

// Note: only for x86. Not sure if simd friendly
static inline u32 uintlog2(const u32 x) {
  uint32_t y;
  asm("\tbsr %1, %0\n" : "=r"(y) : "r"(x));
  return y;
}

constexpr auto square(auto x) { return x * x; }

constexpr u32 fftshiftidx(u32 i, u32 n) {
  return euclid_mod(i + (n + 1) / 2, n);
}

template <typename T>
void leftRotate(std::vector<T>& arr, u32 d) {
  auto n = arr.size();
  d = d % n; // To handle case when d >= n

  // Reverse the first d elements
  std::reverse(arr.begin(), arr.begin() + d);

  // Reverse the remaining elements
  std::reverse(arr.begin() + d, arr.end());

  // Reverse the whole array
  std::reverse(arr.begin(), arr.end());
}

template <typename T>
void fftshift(std::vector<T>& arr) {
  auto n = arr.size();
  u32 d = (n + 1) % 2;
  leftRotate(arr, d);
}

constexpr float annularProfile(float x, float y, float L, float r, float beta) {
  return square(square(L)) /
         (square(x * x + beta * y * y - r * r) + square(square(L)));
}

constexpr auto upow(auto x, u32 N) {
  if (N == 0) {
    return 1;
  }
  if (N == 1) {
    return N;
  }
  return upow(x, N / 2) * upow(x, (N + 1) / 2);
}

template <class T>
struct RangeConf {
  T start;
  T end;
  u64 n;

  constexpr T d() const { return (end - start) / n; }
  constexpr T ith(uint i) const { return start + i * d(); }
};
