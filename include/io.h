#pragma once
#include "Eigen/Dense"
#include "betterexc.h"
#include "hermEigen.h"
#include "mathhelpers.h"
#include "typedefs.h"
#include <fstream>
#include <hdf5.h>
#include <iostream>
#include <ranges>
#include <toml++/toml.hpp>
#include <vector>

static Eigen::IOFormat defaultFormat(Eigen::StreamPrecision,
                                     Eigen::DontAlignCols, " ", "\n", "", "",
                                     "", "");
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

template <size_t n>
void writeArray(std::string s, hid_t fid, void* data, hsize_t sizes[n]) {
  // std::array<hsize_t, sizeof(sizes)> dims{sizes};
  hid_t space = H5Screate_simple(n, sizes, nullptr);
  // hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  hid_t set = H5Dcreate2(fid, s.c_str(), H5T_NATIVE_DOUBLE_g, space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // fid.createDataSet(s, H5::PredType::NATIVE_DOUBLE, space);
  hid_t res =
      H5Dwrite(set, H5T_NATIVE_DOUBLE_g, H5S_ALL, space, H5P_DEFAULT, data);
  if (res < 0) {
    std::cout << "Failed to write HDF5 fid\n";
  }
}

template <size_t n>
void writeSingleArray(std::string s, hid_t fid, void* data, hsize_t sizes[n]) {
  // std::array<hsize_t, sizeof(sizes)> dims{sizes};
  hid_t space = H5Screate_simple(n, sizes, nullptr);
  // hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  hid_t set = H5Dcreate2(fid, s.c_str(), H5T_NATIVE_FLOAT_g, space, H5P_DEFAULT,
                         H5P_DEFAULT, H5P_DEFAULT);
  // fid.createDataSet(s, H5::PredType::NATIVE_DOUBLE, space);
  hid_t res =
      H5Dwrite(set, H5T_NATIVE_FLOAT_g, H5S_ALL, space, H5P_DEFAULT, data);
  if (res < 0) {
    std::cout << "Failed to write HDF5 fid\n";
  }
}

inline bool file_exists(const std::string& name) {
  if (FILE* file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
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

template <class T>
void writeBinary(std::string filename, std::span<T> span) {
  std::ofstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    runtime_exc("Failed to open file: {}", filename);
  }

  file.write(reinterpret_cast<char*>(span.data()), span.size() * sizeof(T));
  file.close();
}

template <class D>
void saveEigen(const std::string& fname, const Eigen::MatrixBase<D>& x) {
  std::ofstream f(fname);
  f << x.format(defaultFormat);
  f.close();
}

template <class T>
Eigen::MatrixX<T> readEigenStream(const std::string& fname) {
  std::string line;
  std::ifstream f(fname);
  u32 m = 0;
  u32 n = 0;
  while (std::getline(f, line)) {
    if (n == 0) {
      auto splits = line | std::ranges::views::split(' ');
      for (const auto& _ : splits) {
        m += 1;
      }
    }
    n += 1;
  }
  f.clear();
  f.seekg(0, std::ios::beg);
  Eigen::MatrixX<T> M(n, m);
  u32 j = 0;
  u32 i = 0;
  M(i, j) >> f;
}

template <class T>
Eigen::MatrixX<T> readEigen(std::string fname) {
  u32 m = 0;
  u32 n = 0;
  std::ifstream f(fname);
  std::vector<std::string> allLines{std::istream_iterator<Line>(f),
                                    std::istream_iterator<Line>()};
  m = allLines.size();
  std::istringstream s(allLines[0]);
  std::vector<T> v{std::istream_iterator<T>(s), std::istream_iterator<T>()};
  n = v.size();
  // f.clear();
  // f.seekg(0, std::ios::beg);
  Eigen::MatrixX<T> M(m, n);
  u32 j = 0;
  for (const auto& line : allLines) {
    std::istringstream stream(line);
    std::vector<T> v{std::istream_iterator<T>(stream),
                     std::istream_iterator<T>()};
    u32 i = 0;
    for (const auto bleh : v) {
      M(j, i) = bleh;
      i += 1;
    }
    j += 1;
  }
  return M;
}

template <class T>
std::vector<T> tarrayToVec(const toml::array& arr) {
  std::vector<T> tmp(arr.size());
  for (u64 i = 0; i < arr.size(); i++) {
    tmp[i] = arr[i].value<T>().value();
  }
  return tmp;
}

RangeConf<Vector2d> tblToVecRange(const toml::table& tbl);
RangeConf<double> tblToRange(toml::table& tbl);

std::optional<EigenSolution> loadDiag(std::string fname);
