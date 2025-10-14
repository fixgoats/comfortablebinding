#pragma once
#include <fstream>
#include "typedefs.h"
#include "betterexc.h"
#include <span>
#include <vector>

struct Line {
  friend std::istream& operator>>(std::istream& is, Line& line) {
    return std::getline(is, line.lineTemp);
  }

  // Output function.
  friend std::ostream& operator<<(std::ostream& os, const Line& line) {
    return os << line.lineTemp;
  }

  // cast to needed result
  operator std::string() const { return lineTemp; }
  // Temporary Local storage for line
  std::string lineTemp{};
};

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

template <class T>
void writeBinary(std::string filename, std::span<T> span) {
  std::ofstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    runtime_exc("Failed to open file: {}", filename);
  }

  file.write(reinterpret_cast<char*>(span.data()), span.size() * sizeof(T));
  file.close();
}

