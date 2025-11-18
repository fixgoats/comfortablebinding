#pragma once
#include "Eigen/Dense"
#include "geometry.h"
#include "mathhelpers.h"
#include "vkcore.h"

using Eigen::VectorXcd, Eigen::VectorXd, Eigen::Vector2d, Eigen::MatrixXcd,
    Eigen::MatrixXd;

typedef std::vector<std::vector<std::pair<f64, u32>>> Delta;

Delta delta(const VectorXd& D, RangeConf<f64> ec, f64 sharpening, f64 cutoff);

VectorXcd planeWave(Vector2d k, const std::vector<Point>& points);
std::vector<f64> disp(const VectorXd& D, const MatrixXcd& UH,
                      const std::vector<Point>& points, f64 lat_const,
                      std::vector<RangeConf<Vector2d>> kc, RangeConf<f64>& ec,
                      f64 sharpening, f64 cutoff, bool printProgress = true);
std::vector<f64> DOS(const VectorXd& D, const MatrixXcd& UH,
                     const std::vector<Point>& points, f64 lat_const,
                     RangeConf<f64> kxc, RangeConf<f64> kyc, RangeConf<f64>& ec,
                     f64 sharpening, f64 cutoff, bool printProgress = true);

std::vector<f32> GPUEsection(Manager& m, const VectorXd& D, const MatrixXcd& UH,
                             const std::vector<Point>& points, f64 lat_const,
                             RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
                             f64 sharpening, f64 cutoff);
std::vector<f64> Esection(const VectorXd& D, const MatrixXcd& UH,
                          const std::vector<Point>& points, f64 lat_const,
                          RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
                          f64 sharpening, f64 cutoff,
                          bool printProgress = true);
std::vector<f64> fullSDF(const VectorXd& D, const MatrixXcd& UH,
                         const std::vector<Point>& points, f64 lat_const,
                         RangeConf<f64> kxc, RangeConf<f64> kyc,
                         RangeConf<f64>& ec, f64 sharpening, f64 cutoff,
                         bool printProgress = true);

template <class Func>
MatrixXd finite_hamiltonian(u32 n_points, const std::vector<Neighbour>& nbs,
                            Func f) {
  MatrixXd H = MatrixXd::Zero(n_points, n_points);
  for (const auto& nb : nbs) {
    f64 val = f(nb.d);
    H(nb.i, nb.j) = val;
    H(nb.j, nb.i) = val;
  }
  return H;
}
MatrixXd pointsToFiniteHamiltonian(const std::vector<Point>& points,
                                   const kdt::KDTree<Point>& kdtree,
                                   f64 radius);
