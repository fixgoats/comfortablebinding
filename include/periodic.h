#pragma once
#include "Eigen/Dense"
#include "geometry.h"
#include "hermEigen.h"
#include "typedefs.h"
#include <mdspan>
#include <toml++/toml.hpp>

using Eigen::MatrixXcd, Eigen::Vector2d;

struct PerConf {
  std::string fname;
  std::vector<Point> points;
  std::vector<Neighbour> nbs;
  std::optional<RangeConf<double>> kxrange;
  std::optional<RangeConf<double>> kyrange;
  std::vector<RangeConf<Vector2d>> kpath;
  Vector2d lat_vecs[2];
  Vector2d dual_vecs[2];
  bool do2D; // combine dispersion and dos, since dos follows from 2d disp
  bool doPath;

  // lat_vecs needs to be populated before running this. Should only be ran
  // once.
  void parseConnections(toml::array& arr) {
    nbs.reserve(arr.size() * 3); // decent guess for number of connections
    for (const auto& st : arr) {
      auto tbl = *st.as_table();
      size_t i = tbl["source"].value<size_t>().value();
      auto dests = *tbl["destinations"].as_array();
      for (const auto& dest : dests) {
        const auto t = *dest.as_array();
        s64 n0 = t[1].value<s64>().value();
        s64 n1 = t[2].value<s64>().value();
        size_t j = t[0].value<size_t>().value();
        Vector2d d = points[j].asVec() + n0 * lat_vecs[0] + n1 * lat_vecs[1] -
                     points[i].asVec();
        nbs.emplace_back(i, j, d);
      }
    }
    nbs.shrink_to_fit();
  }

  void makeDuals() {
    dual_vecs[0] = {
        2 * M_PI /
            (lat_vecs[0][0] - lat_vecs[0][1] * lat_vecs[1][0] / lat_vecs[1][1]),
        2 * M_PI /
            (lat_vecs[0][1] -
             lat_vecs[0][0] * lat_vecs[1][1] / lat_vecs[1][0])};
    dual_vecs[1] = {
        2 * M_PI /
            (lat_vecs[1][0] - lat_vecs[1][1] * lat_vecs[0][0] / lat_vecs[0][1]),
        2 * M_PI /
            (lat_vecs[1][1] -
             lat_vecs[1][0] * lat_vecs[0][1] / lat_vecs[0][0])};
  }
};

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

std::optional<PerConf> tomlToPerConf(std::string tomlPath);
int doPeriodicModel(const PerConf& conf);
