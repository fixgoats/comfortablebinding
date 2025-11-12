#pragma once
#include "Eigen/Dense"
#include "geometry.h"
#include "hermEigen.h"
#include "typedefs.h"
#include <mdspan>

using Eigen::MatrixXcd, Eigen::Vector2d;

template <class Func>
void update_hamiltonian(MatrixXcd& H, const std::vector<Neighbour>& nbs,
                        Vector2d k, Func f, bool reset = false) {
  if (reset) {
    for (const auto& nb : nbs) {
      H(nb.i, nb.j) = 0;
      H(nb.j, nb.i) = 0;
    }
  }
  for (const auto& nb : nbs) {
    c64 val = f(nb.d) * std::exp(c64{0, k.dot(nb.d)});
    H(nb.i, nb.j) += val;
    H(nb.j, nb.i) += std::conj(val);
  }
}

template <class Func>
void disp2d(RangeConf<f64> kxrange, RangeConf<f64> kyrange, MatrixXcd& H,
            const std::vector<Neighbour>& nbs, Func f) {
  std::vector<double> disprel(kxrange.n * kyrange.n * H.rows());
  auto energy_view =
      std::mdspan(disprel.data(), kyrange.n, kxrange.n, H.rows());
  for (u32 j = 0; j < kyrange.n; j++) {
    f64 ky = kyrange.ith(j);
    for (u32 i = 0; i < kxrange.n; i++) {
      f64 kx = kxrange.ith(i);
      update_hamiltonian(H, nbs, {kx, ky}, f, i | j);
      EigenSolution eigsol = hermitianEigenSolver(H);
      for (u32 k = 0; k < nbs.size(); k++) {
        energy_view[i, j, k] = eigsol.D(k);
      }
    }
  }
}

template <class Func>
void disp1d(RangeConf<f64> krange, MatrixXcd& H,
            const std::vector<Neighbour>& nbs, Func f, bool axis) {
  std::vector<double> disprel(krange.n * H.rows());
  auto energy_view = std::mdspan(disprel.data(), krange.n, H.rows());
  for (u32 j = 0; j < krange.n; j++) {
    f64 k = krange.ith(j);
    update_hamiltonian(H, nbs, {!axis * k, !axis * k}, f, j);
    EigenSolution eigsol = hermitianEigenSolver(H);
    for (u32 k = 0; k < nbs.size(); k++) {
      energy_view[j, k] = eigsol.D(k);
    }
  }
}

template <class Func>
void disppath(RangeConf<Vector2d> krange, MatrixXcd& H,
              const std::vector<Neighbour>& nbs, Func f) {
  std::vector<double> disprel(krange.n * H.rows());
  auto energy_view = std::mdspan(disprel.data(), krange.n, H.rows());
  for (u32 j = 0; j < krange.n; j++) {
    auto k = krange.ith(j);
    update_hamiltonian(H, nbs, k, f, j);
    EigenSolution eigsol = hermitianEigenSolver(H);
    for (u32 k = 0; k < nbs.size(); k++) {
      energy_view[j, k] = eigsol.D(k);
    }
  }
}
