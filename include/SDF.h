#pragma once
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "geometry.h"
#include "mathhelpers.h"
#include <iostream>
// #include "vkcore.h"

using Eigen::VectorXcd, Eigen::VectorXd, Eigen::Vector2d, Eigen::MatrixXcd,
    Eigen::MatrixXd, Eigen::SparseMatrix, Eigen::Vector2i;

typedef std::vector<std::vector<std::pair<f64, u32>>> Delta;

inline MatrixXd DenseH(const std::vector<Point>& points,
                       const kdt::KDTree<Point>& kdtree, double radius,
                       f64 (*f)(Vector2d)) {

  std::vector<Neighbour> nb_info = pointsToNbs(points, kdtree, radius);
  MatrixXd H = MatrixXd::Zero(points.size(), points.size());
  for (const auto& nb : nb_info) {
    f64 val = f(nb.d);
    H(nb.i, nb.j) = val;
    H(nb.j, nb.i) = val;
  }
  return H;
}

template <class Func>
inline SparseMatrix<c64> SparseHC(const std::vector<Point>& points,
                                  const kdt::KDTree<Point>& kdtree, f64 radius,
                                  Func f) {

  std::vector<Neighbour> nb_info = pointsToNbs(points, kdtree, radius);
  SparseMatrix<c64> H(points.size(), points.size());
  for (const auto& nb : nb_info) {
    c64 val = f(nb.d);
    H.insert(nb.i, nb.j) = val;
    H.insert(nb.j, nb.i) = std::conj(val);
  }
  return H;
}

template <class Func>
inline SparseMatrix<c64> SparseHC(const Eigen::MatrixX2d& points,
                                  const Eigen::MatrixX2i& couplings, Func f) {

  SparseMatrix<c64> H(points.rows(), points.rows());
  for (s64 i = 0; i < couplings.rows(); ++i) {
    std::cout << couplings(i, 0) << '\n';
    std::cout << couplings(i, 1) << '\n';
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    c64 val = f(d);
    H.insert(couplings(i, 0), couplings(i, 1)) = val;
    H.insert(couplings(i, 1), couplings(i, 0)) = std::conj(val);
  }
  return H;
}

template <class Func>
inline SparseMatrix<c64> SparseC(const Eigen::MatrixX2d& points,
                                 const Eigen::MatrixX2i& couplings, Func f) {
  SparseMatrix<c64> H(points.rows(), points.rows());
  for (s64 i = 0; i < couplings.rows(); ++i) {
    std::cout << couplings(i, 0) << '\n';
    std::cout << couplings(i, 1) << '\n';
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    c64 val = f(d);
    H.insert(couplings(i, 0), couplings(i, 1)) = val;
    H.insert(couplings(i, 1), couplings(i, 0)) = val;
  }
  return H;
}

inline SparseMatrix<f64> SparseH(const std::vector<Point>& points,
                                 const kdt::KDTree<Point>& kdtree, f64 radius,
                                 f64 (*f)(Vector2d)) {

  std::vector<Neighbour> nb_info = pointsToNbs(points, kdtree, radius);
  SparseMatrix<double> H(points.size(), points.size());
  for (const auto& nb : nb_info) {
    f64 val = f(nb.d);
    H.insert(nb.i, nb.j) = val;
    H.insert(nb.j, nb.i) = val;
  }
  return H;
}

struct SdfConf {
  // sharpness of Gaussian used to approximate Delta function
  double sharpening;
  // values below this will be removed from the Delta function.
  double cutoff;
  // In units of average nearest neighbour distance.
  double searchRadius;
  // if hasValue, do a dispersion relation with the line given by this range.
  std::vector<RangeConf<Vector2d>> kpath;
  // Set Emin=Emax to automatically set E range
  std::optional<RangeConf<double>> dispE;
  std::optional<std::string> saveDiagonalisation;
  std::optional<std::string> useSavedDiag;
  std::optional<std::string> saveHamiltonian;
  // in units of "Brillouin zone", i.e. 1 = 2pi/a where a is a lattice constant.
  // Average nearest neighbour distance is used as a proxy for a
  RangeConf<double> sectionKx;
  RangeConf<double> sectionKy;
  RangeConf<double> SDFKx;
  RangeConf<double> SDFKy;
  // Set Emin=Emax to automatically set E range
  RangeConf<double> SDFE;
  // lattice point input (could possibly take special strings to do common
  // lattices)
  double fixed_e;
  std::string pointPath;
  // Output file name
  std::string H5Filename;
  // Generally what I want. Not any more expensive than computing DOS and the
  // rest can be obtained by slicing this dataset.
  bool doFullSDF;
  // Only set if you don't want to output a full sdf to disk. Will be ignored if
  // doFullSDF is set.
  bool doDOS;
  bool doEsection;
  bool doPath;
};

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

// std::vector<f32> GPUEsection(Manager& m, const VectorXd& D, const MatrixXcd&
// UH,
//                              const std::vector<Point>& points, f64 lat_const,
//                              RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
//                              f64 sharpening, f64 cutoff);
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

std::optional<SdfConf> tomlToSdfConf(const std::string& tbl);
int doSDFcalcs(SdfConf& conf);
