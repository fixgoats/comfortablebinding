#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "geometry.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include "spdlog/spdlog.h"
#include "vkcore.hpp"
#include <cxxopts.hpp>
#include <gsl/gsl_sf.h>
#include <iostream>
#include <random>
#include <vector>

using Eigen::Vector2cd, Eigen::Vector2cf, Eigen::MatrixX2d, Eigen::MatrixX2f,
    Eigen::MatrixX2i, Eigen::MatrixXcf, Eigen::VectorXcf;

struct SpecConsts {
  u32 len;
  u32 wave_size;
  f32 dt;
  f32 alpha;
};

auto hankelDD(const SparseMatrix<c64>& J, f64 p, f64 alpha) {
  return [&, p, alpha](Eigen::VectorXcd psi) {
    return p * psi - c64{1, alpha} * psi.cwiseAbs2().cwiseProduct(psi) +
           J * psi;
  };
}

template <class F, class State>
State rk4step(const State& x, f64 dt, F rhs) {
  spdlog::debug("Function: rk4step.");
  State k1 = rhs(x);
  State k2 = rhs(x + 0.5 * dt * k1);
  State k3 = rhs(x + 0.5 * dt * k2);
  State k4 = rhs(x + dt * k3);
  State ret = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
  spdlog::debug("Exiting rk4step.");
  return ret;
}

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

  return 0;
}
