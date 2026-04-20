#pragma once
#include "Eigen/Dense"
#include "spdlog/spdlog.h"
#include "typedefs.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>
#ifndef NDEBUG
#include <cassert>
#endif // !DEBUG

using Eigen::MatrixXcd;

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

constexpr s32 euclid_mod(s32 a, s32 b) {
  s32 tmp = a % b;
  return tmp < 0 ? tmp + b : tmp;
}

constexpr bool fleq(auto x, auto y, double tol) {
  return std::abs(y - x) < tol;
}

constexpr s64 binpow(s64 base, s64 exp) {
  s64 ret = 1;
  while (exp > 0) {
    if (static_cast<bool>(exp & 1)) {
      ret = ret * base;
    }
    base = base * base;
    exp >>= 1;
  }
  return ret;
}
// Note: only for x86. Not sure if simd friendly
static inline u32 uintlog2(u32 x) {
  uint32_t y;
  asm("\tbsr %1, %0\n" : "=r"(y) : "r"(x));
  return y;
}

static inline u32 uintlog2n(u32 x, u32 n) {
  uint32_t y;
  asm("\tbsr %1, %0\n" : "=r"(y) : "r"(x));
  y /= n;
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

template <class T>
constexpr size_t vecBytes(std::vector<T>& v) {
  return v.size() * sizeof(T);
}

template <class T>
Eigen::VectorX<T> linspace(RangeConf<T> rc, bool endpoint) {
  u64 oneOrZero = endpoint ? 1 : 0;
  Eigen::VectorX<T> lin(rc.n + oneOrZero);
  for (u64 i = 0; i < rc.n + oneOrZero; ++i) {
    lin(i) = rc.ith(i);
  }
  return lin;
}

template <class T>
struct COOMat {
  std::unordered_map<size_t, std::unordered_map<size_t, T>> elements;
  u64 rows;
  u64 cols;

  template <class Int>
  T& operator()(Int i, Int j) {
    return elements[i][j];
  }

  template <class Int>
  T operator()(Int i, Int j) const {
    return elements[i][j];
  }
};

template <class T>
struct CSRMat {
  std::vector<size_t> row_indices;
  std::vector<size_t> col_indices;
  std::vector<T> data;
  u64 rows;
  u64 cols;

  CSRMat<T>() = default;
  CSRMat<T>(const COOMat<T>& other) {
    spdlog::debug("converting COOMat to CSR.");
    u64 data_size = 0;
    for (const auto& map : other.elements) {
      data_size += map.second.size();
    }
    rows = other.rows;
    cols = other.cols;

    row_indices.resize(rows + 1);
    col_indices.resize(data_size);
    data.resize(data_size);
    u64 row = 0;
    u64 total = 0;
    for (u64 i = 0; i < rows; ++i) {
      row_indices[i] = total;
      if (other.elements.contains(i)) {
        for (const auto& spair : other.elements.at(i)) {
          col_indices[total] = spair.first;
          data[total] = spair.second;
          ++total;
        }
      }
      ++row;
    }
    row_indices[row] = total;
  }
};

template <class A, class B, class C, class D>
void spmv(CSRMat<A> M, B a, C* v, D b, C* out) {
  spdlog::debug("function spmv const.");
  u64 nrows = M.row_indices.size() - 1;
#pragma omp parallel for
  for (u64 i = 0; i < nrows; ++i) {
    u64 row_start = M.row_indices[i];
    u64 row_end = M.row_indices[i + 1];
    out[i] = {};
    for (u64 j = row_start; j < row_end; ++j) {
      out[i] += a * M.data[j] * v[M.col_indices[j]] + b * out[i];
    }
  }
}

template <class A, class B, class C, class D>
void spmv(CSRMat<A> M, B a, C* v, D* b, C* out) {
  spdlog::debug("function spmv array.");
  u64 nrows = M.row_indices.size() - 1;
#pragma omp parallel for
  for (u64 i = 0; i < nrows; ++i) {
    u64 row_start = M.row_indices[i];
    u64 row_end = M.row_indices[i + 1];
    out[i] = {};
    for (u64 j = row_start; j < row_end; ++j) {
      out[i] += a * M.data[j] * v[M.col_indices[j]] + b[i] * out[i];
    }
  }
}

inline MatrixXcd csr_to_dense(CSRMat<c64> M) {
  spdlog::debug("Function csr_to_dense.");
  MatrixXcd H = MatrixXcd::Zero(M.rows, M.cols);
  spdlog::debug("Made matrix");

#pragma omp parallel for
  for (u64 i = 0; i < M.rows; ++i) {
    spdlog::debug("Writing to row", i);
    u64 row_start = M.row_indices[i];
    u64 row_end = M.row_indices[i + 1];
    for (u64 j = row_start; j < row_end; ++j) {
      spdlog::debug("Writing element", M.data[j].real(), M.data[j].imag(),
                    "to row", j, "col", M.col_indices[j]);
      H(i, M.col_indices[j]) = M.data[j];
    }
  }
  return H;
}

template <class Func, class A, class... Args>
void apply(Func f, u64 n, A* __restrict__ dest, Args... a) {
  spdlog::debug("Function template apply.");
#pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    dest[i] = f(a[i]...);
  }
}
