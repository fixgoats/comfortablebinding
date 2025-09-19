#pragma once
#include "betterexc.h"
#include "metaprogramming.h"
#include "typedefs.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <type_traits>
#include <vector>

constexpr f32 hbar = 6.582119569e-1;
constexpr f32 muB = 5.7883818060e-2;
constexpr f32 echarge = 1e3;

using std::bit_cast;

template <typename T>
constexpr u32 euclid_mod(T a, u32 b) {
  assert(b != 0);
  return a % b;
}

template <class T>
constexpr auto numfmt(T x) {
  if constexpr (std::is_same_v<T, c32> or std::is_same_v<T, c64>) {
    return std::format("({}+{}j)", x.real(), x.imag());
  } else {
    return std::format("{}", x);
  }
}

template <class T>
void writeCsv(std::ofstream& of, const std::vector<T>& v, u32 nColumns,
              u32 nRows = 1, u32 stride = 1, u32 offset = 0,
              const std::vector<std::string>& heading = {}) {
  if (offset + stride * nColumns * nRows < v.size()) {
    runtime_exc("There aren't this many elements in the vector.");
  }
  std::string out;
  if (heading.size()) {
    for (const auto& h : heading) {
      of << h << ' ';
    }
    of << '\n';
  }
  for (u32 j = 0; j < nRows; j++) {
    for (u32 i = 0; i < nColumns; i += stride) {
      of << v[j * nColumns * stride + i + offset] << ' ';
    }
    of << '\n';
  }
  of.close();
}

template <>
void writeCsv<c32>(std::ofstream& of, const std::vector<c32>& v, u32 nColumns,
                   u32 nRows, u32 stride, u32 offset,
                   const std::vector<std::string>& heading);
template <>
void writeCsv<c64>(std::ofstream& of, const std::vector<c64>& v, u32 nColumns,
                   u32 nRows, u32 stride, u32 offset,
                   const std::vector<std::string>& heading);

template <class T>
void writeBinary(std::string filename, std::span<T> span) {
  std::ofstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    runtime_exc("Failed to open file: {}", filename);
  }

  file.write(reinterpret_cast<char*>(span.data()), span.size() * sizeof(T));
  file.close();
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

template <typename T>
u8 mapToColor(T v, T min, T max) {
  return static_cast<u8>(256 * (v - min) / (max - min));
}

template <typename InputIt, typename OutputIt>
void colorMap(InputIt it, InputIt end, OutputIt out) {
  const auto max = *std::max_element(it, end);
  const auto min = *std::min_element(it, end);
  std::transform(it, end, out, [&](auto x) { return mapToColor(x, min, max); });
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

template <typename InputIt>
std::vector<u8> colorMapVec(InputIt it, InputIt end) {
  std::vector<u8> out(end - it);
  colorMap(it, end, out.begin());
  return out;
}
