#include "Eigen/Dense"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "typedefs.h"
#include <cxxopts.hpp>
#include <iostream>
#include <random>
#include <ranges>
#include <sstream>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixX2d;
constexpr u32 Nx = 40;
constexpr u32 Ny = 40;
static Eigen::IOFormat defaultFormat(Eigen::StreamPrecision,
                                     Eigen::DontAlignCols, " ", "\n", "", "",
                                     "", "");

MatrixXd couplingmat(VectorXd xs, VectorXd ys, f64 rsq) {
  const u32 n = xs.size();
  std::cout << n << '\n';
  MatrixXd J(n, n);
  for (u32 j = 0; j < n - 1; j++) {
    for (u32 i = j + 1; i < n; i++) {
      if (square(xs[i] - xs[j]) + square(ys[i] - ys[j]) < rsq) {
        J(j, i) = 1;
        J(i, j) = 1;
      } else {
        J(j, i) = 0;
        J(i, j) = 0;
      }
    }
  }
  return J;
}

/*
template <typename Out>
void split(const std::string& s, char delim, Out result) {
  std::istringstream iss(s);
  std::string item;
  while (std::getline(iss, item, delim)) {
    *result++ = item;
  }
}

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}
*/

/// nx: number of hexagons along x axis.
/// ny: number of hexagons along y axis.
/// a: lattice constant.
void makeHexLattice(u32 nx, u32 ny, f64 a) {
  Eigen::Vector2d v1{a * std::cos(M_2_PI / 3), a * std::cos(M_2_PI / 3)};
  Eigen::Vector2d v2{a * std::cos(M_PI / 3), a * std::cos(M_PI / 3)};
  Eigen::Vector2d h{a, 0};
  u32 npoints = 2 * (2 * ny + 1) + (nx - 1) * (2 * ny + 2);

  // Eigen::MatrixX2d = ;
  Eigen::MatrixX2d H(npoints, 2);
  if (nx % 2 == 1) {
    u32 idx = 0;
    for (u32 i = 0; i < 2 * ny + 1; i++) {
      H(idx, {1, 2}) = (i + 1 / 2) * v1 + (i / 2) * v2;
      idx += 1;
    }

    for (u32 j = 1; j < nx; j++) {
      for (u32 i = 0; i < 2 * ny + 1; i++) {
        H(idx, {1, 2}) = (i + 1 / 2) * v1 + (i / 2) * v2;
        idx += 1;
      }
    }
  }
  H(0, {1, 2}) = 0 * v1 + 0 * v2;
  H(1, {1, 2}) = 1 * v1 + 0 * v2;
  H(2, {1, 2}) = 1 * v1 + 1 * v2;
  /*if (nx % 2 == 0) {
    for (u32 i = 0; i < nx + 1; i++) {
      H(2 * i * (2 * ny + 1), {1, 2}) = 2 * i * h + (i / 2) * v2;
      for (u32 j = 0; j < ny; j++) {
        H(2 * i * (2 * ny + 1) + 2 * j, {1, 2}) = (i / 2) * v2
      }
    }
    for (u32 i = 0; i < nx; i++) {
      H(i * (2 * ny + 1), {1, 2});
      for (u32 j = 0; j < ny; j++) {
      }
    }
  }*/
}

MatrixXd couplingmat(MatrixX2d points, f64 rsq) {
  const u32 n = points.rows();
  std::cout << n << '\n';
  MatrixXd J(n, n);
  for (u32 j = 0; j < n - 1; j++) {
    for (u32 i = j + 1; i < n; i++) {
      const Eigen::Vector2d x = points(i, {0, 1}) - points(j, {0, 1});
      if (x.squaredNorm() < rsq) {
        J(j, i) = 1;
        J(i, j) = 1;
      } else {
        J(j, i) = 0;
        J(i, j) = 0;
      }
    }
  }
  return J;
}

template <class D>
void saveEigen(std::string fname, Eigen::MatrixBase<D>& x) {
  std::ofstream f(fname);
  f << x.format(defaultFormat);
  f.close();
}

/* template <class T>
Eigen::MatrixX<T> readEigen(std::string fname) {
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
  while (std::getline(f, line)) {
    auto x = split(line, ' ');
    for (u32 i = 0; i < m; i++) {
      if (std::is_same_v<f64, T>)
        M(j, i) = std::stod(x[i]);
      if (std::is_same_v<f32, T>)
        M(j, i) = std::stof(x[i]);
    }
    j += 1;
  }
  return M;
} */

// c64 stocd(std::string s) { auto splits = s | std::ranges::views::split('+');
// }

template <class T>
Eigen::MatrixX<T> readEigenStream(std::string fname) {
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
      std::cout << bleh << ' ' << M(j, i) << '\n';
      i += 1;
    }
    j += 1;
  }
  return M;
}

/* int main(const int argc, const char* const* argv) {
  output<f64>("../data/testfile.txt");
  output<c64>("../data/complexfile.txt");
  auto m = readEigen<f64>("../data/testfile.txt");
  std::cout << m << std::endl;
} */

int main(const int argc, const char* const* argv) {
  cxxopts::Options options("MyProgram", "bleh");
  options.add_options()("p,points", "File name", cxxopts::value<std::string>())(
      "c,chain", "Make chain points", cxxopts::value<u32>())(
      "r,radius", "Maximum separation between points to consider",
      cxxopts::value<f64>());

  auto result = options.parse(argc, argv);
  if (result["c"].count()) {
    u32 n = result["c"].as<u32>();
    MatrixX2d points(n, 2);
    for (u32 i = 0; i < n; i++) {
      points(i, {0, 1}) = Eigen::Vector2d{i, 0};
    }
    saveEigen("chain.txt", points);
  }
  if (result["p"].count() && result["r"].count()) {
    MatrixXd m = readEigen<f64>(result["p"].as<std::string>());
    MatrixXd H = couplingmat(m, result["r"].as<f64>());

    Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(H);

    if (eigensolver.info() != Eigen::Success) {
      std::cout << eigensolver.info() << std::endl;
      abort();
    }
    std::ofstream f("eigvals.txt");
    f << eigensolver.eigenvalues().format(defaultFormat);
    f.close();
  }
  if (result["s"].count()) {
    MatrixXd m = readEigen<f64>(result["p"].as<std::string>());
    MatrixXd H = couplingmat(m, result["r"].as<f64>());

    Eigen::VectorXcd psi;
    if (result["i"].count()) {
      psi = readEigen<c64>(result["i"].as<std::string>());
    } else {
      std::random_device dev;
      std::mt19937_64 mt(dev());
      std::uniform_real_distribution<f64> dis(0, M_2_PI);
      for (u32 i = 0; i < H.rows(); i++) {
        psi(i) = c64{cos(dis(mt)), sin(dis(mt))};
      }
    }
  }
}
