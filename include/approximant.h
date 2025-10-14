#pragma once
#include "geometry.h"
#include "Eigen/Dense"
using Eigen::VectorXcd, Eigen::VectorXd, Eigen::Vector2d, Eigen::MatrixXcd;

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


