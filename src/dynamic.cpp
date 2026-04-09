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
  if (tbl.contains(#key))                                                      \
  c.key = *tbl[#key].value<decltype(c.key)>()

#undef SET_STRUCT_FIELD

auto basicNonLin(const SparseMatrix<c64>& iH, f64 alpha) {
  return [&iH, alpha](const VectorXcd& x) {
    return VectorXcd(iH * x + alpha * x.cwiseAbs2());
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

auto coupledNonLin(const SparseMatrix<c64>& iH, const VectorXd& P, f64 alpha,
                   f64 R, f64 gamma, f64 Gamma) {
  return [&, alpha, R, gamma, Gamma](const StateAndReservoir& x) {
    return StateAndReservoir{iH * x.x + alpha * x.x.cwiseAbs2() + R * x.n -
                                 gamma * x.x,
                             P - Gamma * x.n};
  };
}

auto tetmNonLin(const SparseMatrix<f64>& iJ, const SparseMatrix<c64>& iL, f64 p,
                f64 alpha) {
  return [&, p, alpha](const MatrixX2cd& psi) {
    spdlog::debug("Function: tetmNonLin.");
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
    MatrixX2cd third = iJ * psi;
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
    spdlog::debug("Exiting tetmNonLin.");
    return first + second + third + fourth;
  };
}

MatrixX2cd explTETM(MatrixX2cd psi, const SparseMatrix<c64>& J,
                    const SparseMatrix<c64>& L, f64 p, f64 alpha) {
  MatrixX2cd complicated(psi.rows(), 2);
  complicated.col(0) = L * psi.col(1);
  complicated.col(1) = -L.conjugate() * psi.col(0);
  return p * psi - c64{1, alpha} * psi.cwiseAbs2().cwiseProduct(psi) + J * psi +
         complicated;
  // +complicated;
  // iJ * psi;
  // - c64{0, alpha} *  +
  //        iJ * psi + complicated;
}

MatrixX2cd tetmRK4Step(MatrixX2cd psi, const SparseMatrix<c64>& J,
                       const SparseMatrix<c64>& L, f64 p, f64 alpha, f64 dt) {

  MatrixX2cd k1 = explTETM(psi, J, L, p, alpha);
  MatrixX2cd k2 = explTETM(psi + 0.5 * dt * k1, J, L, p, alpha);
  MatrixX2cd k3 = explTETM(psi + 0.5 * dt * k2, J, L, p, alpha);
  MatrixX2cd k4 = explTETM(psi + dt * k3, J, L, p, alpha);
  MatrixX2cd ret = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
  return ret;
}
