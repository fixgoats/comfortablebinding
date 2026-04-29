#include "dynamic.h"
#include "Eigen/Core"
#include "SDF.h"
#include "geometry.h"
#include "hermEigen.h"
#include "sparse.hpp"
#include "spdlog/spdlog.h"
#include <complex>
#include <cstddef>
#include <cstring>
#include <print>
#include <toml++/toml.hpp>

using Eigen::MatrixXd, Eigen::VectorXcd, Eigen::MatrixX2cd, Eigen::VectorXcf,
    Eigen::MatrixXcf;

#define SET_STRUCT_FIELD(c, tbl, key)                                          \
  if ((tbl).contains(#key))                                                    \
  (c).key = *(tbl)[#key].value<decltype((c).key)>()

#undef SET_STRUCT_FIELD

auto basic_non_lin(const SparseMatrix<c64>& i_h, f64 alpha) {
  return [&i_h, alpha](const VectorXcd& x) {
    return VectorXcd(i_h * x + alpha * x.cwiseAbs2());
  };
}

struct StateAndReservoir {
  typedef StateAndReservoir Self;
  VectorXcd x;
  VectorXd n;

  // StateAndReservoir operator*(const f64& b) const;
  Self& operator*=(f64 b) {
    x *= b;
    n *= b;
    return *this;
  }
  friend Self operator*(Self lhs, f64 b) { return lhs *= b; }
  Self& operator+=(const Self& rhs) {
    x += rhs.x;
    n += rhs.n;
    return *this;
  }
  friend Self operator+(Self lhs, const Self& b) { return lhs += b; }
  Self& operator-=(const Self& rhs) {
    x -= rhs.x;
    n -= rhs.n;
    return *this;
  }
  friend Self operator-(Self lhs, const Self& rhs) { return lhs -= rhs; }
};

auto coupled_non_lin(const SparseMatrix<c64>& i_h, const VectorXd& p, f64 alpha,
                     f64 r, f64 gamma, f64 exc_gamma) {
  return [&, alpha, r, gamma, exc_gamma](const StateAndReservoir& x) {
    return StateAndReservoir{i_h * x.x + alpha * x.x.cwiseAbs2() + r * x.n -
                                 gamma * x.x,
                             p - exc_gamma * x.n};
  };
}

auto tetm_non_lin(const SparseMatrix<f64>& i_j, const SparseMatrix<c64>& iL,
                  f64 p, f64 alpha) {
  return [&, p, alpha](const MatrixX2cd& psi) {
    spdlog::debug("Function: tetm_non_lin.");
    spdlog::debug("Making first matrix");
    MatrixX2cd first = (p - 1) * psi;
    u32 n = first.rows();
    u32 m = first.cols();
    spdlog::debug("First matrix has dims {}x{}", n, m);
    spdlog::debug("Making second matrix");
    MatrixX2cd second = -c64{0, alpha} * psi.cwiseAbs2().cwiseProduct(psi);
    n = second.rows();
    m = second.cols();
    spdlog::debug("Second matrix has dims {}x{}", n, m);
    spdlog::debug("Making third matrix");
    MatrixX2cd third = i_j * psi;
    n = third.rows();
    m = third.cols();
    spdlog::debug("Third matrix has dims {}x{}", n, m);
    spdlog::debug("Making fourth matrix");
    MatrixX2cd fourth(psi.rows(), 2);
    fourth.col(0) = iL * psi.col(1);
    fourth.col(1) = -iL.conjugate() * psi.col(0);
    n = fourth.rows();
    m = fourth.cols();
    spdlog::debug("Fourth matrix has dims {}x{}", n, m);
    spdlog::debug("Exiting tetm_non_lin.");
    return first + second + third + fourth;
  };
}

MatrixX2cd expl_tetm(MatrixX2cd psi, const SparseMatrix<c64>& coupling_mat,
                     const SparseMatrix<c64>& tetm_coupling_mat, f64 p,
                     f64 alpha) {
  MatrixX2cd complicated(psi.rows(), 2);
  complicated.col(0) = tetm_coupling_mat * psi.col(1);
  complicated.col(1) = -tetm_coupling_mat.conjugate() * psi.col(0);
  return p * psi - c64{1, alpha} * psi.cwiseAbs2().cwiseProduct(psi) +
         coupling_mat * psi + complicated;
  // +complicated;
  // i_j * psi;
  // - c64{0, alpha} *  +
  //        i_j * psi + complicated;
}

MatrixX2cd tetmRK4Step(MatrixX2cd psi, const SparseMatrix<c64>& coupling_mat,
                       const SparseMatrix<c64>& tetm_coupling_mat, f64 p,
                       f64 alpha, f64 dt) {

  MatrixX2cd k1 = expl_tetm(psi, coupling_mat, tetm_coupling_mat, p, alpha);
  MatrixX2cd k2 =
      expl_tetm(psi + 0.5 * dt * k1, coupling_mat, tetm_coupling_mat, p, alpha);
  MatrixX2cd k3 =
      expl_tetm(psi + 0.5 * dt * k2, coupling_mat, tetm_coupling_mat, p, alpha);
  MatrixX2cd k4 =
      expl_tetm(psi + dt * k3, coupling_mat, tetm_coupling_mat, p, alpha);
  MatrixX2cd ret = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
  return ret;
}
