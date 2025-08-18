#include "Eigen/Dense"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "typedefs.h"
#include <cxxopts.hpp>
#include <iostream>
#include <ranges>

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

template <class T>
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
      M(j, i) = std::stod(x[i]);
    }
    j += 1;
  }
  return M;
}

int main(const int argc, const char* const* argv) {
  cxxopts::Options options("MyProgram", "bleh");
  options.add_options()("p,points", "File name", cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);
  MatrixXd m = readEigen<f64>(result["p"].as<std::string>());
  std::cout << m;
  // std::cout << m << std::endl;
  /*VectorXd xs(Nx * Ny);
  for (u32 j = 0; j < Ny; j++) {
    for (u32 i = 0; i < Nx; i++) {
      xs[Nx * j + i] = i;
    }
  }

  VectorXd ys(Nx * Ny);
  for (u32 j = 0; j < Ny; j++) {
    for (u32 i = 0; i < Nx; i++) {
      ys[Nx * j + i] = j;
    }
  }

  MatrixXd H = couplingmat(xs, ys, 1.1);
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(H);

  if (eigensolver.info() != Eigen::Success) {
    std::cout << eigensolver.info() << std::endl;
    abort();
  }
  std::ofstream f("eigvals.txt");
  f << eigensolver.eigenvalues().format(defaultFormat);
  f.close();*/
}
