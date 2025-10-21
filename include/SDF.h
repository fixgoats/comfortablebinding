#pragma once
#include "mathhelpers.h"
#include "geometry.h"
#include "Eigen/Dense"
using Eigen::VectorXcd, Eigen::VectorXd, Eigen::Vector2d, Eigen::MatrixXcd, Eigen::MatrixXd;

typedef std::vector<std::vector<std::pair<double, u32>>> Delta;

Delta delta(const VectorXd& D, RangeConf<double> ec, double sharpening, double cutoff);

VectorXcd planeWave(Vector2d k, const std::vector<Point>& points);
std::vector<double> disp(const VectorXd& D, const MatrixXcd& UH,
                         const std::vector<Point>& points, double lat_const,
                         RangeConf<Vector2d> kc, RangeConf<double>& ec, double sharpening, double cutoff,
                         bool printProgress = true);
std::vector<double> DOS(const VectorXd& D, const MatrixXcd& UH,
                        const std::vector<Point>& points, double lat_const, RangeConf<double> kxc,
                        RangeConf<double> kyc, RangeConf<double>& ec, double sharpening, double cutoff,
                        bool printProgress = true);

std::vector<double> Esection(const VectorXd& D, const MatrixXcd& UH,
                            const std::vector<Point>& points, double lat_const,
                            RangeConf<double> kxc, RangeConf<double> kyc,
                            double e, double sharpening, double cutoff, bool printProgress = true);
std::vector<double> fullSDF(const VectorXd& D, const MatrixXcd& UH,
                            const std::vector<Point>& points, double lat_const,
                            RangeConf<double> kxc, RangeConf<double> kyc,
                            RangeConf<double>& ec, double sharpening, double cutoff, bool printProgress = true);

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
