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

struct SdfConf {
  // sharpness of Gaussian used to approximate Delta function
  double sharpening;
  // values below this will be removed from the Delta function.
  double cutoff;
  // In units of average nearest neighbour distance.
  double search_radius;
  // if hasValue, do a dispersion relation with the line given by this range.
  std::vector<RangeConf<Vector2d>> kpath;
  // Set Emin=Emax to automatically set E range
  std::optional<RangeConf<double>> disp_e;
  std::optional<std::string> save_diag;
  std::optional<std::string> use_saved_diag;
  std::optional<std::string> save_hamiltonian;
  // in units of "Brillouin zone", i.e. 1 = 2pi/a where a is a lattice constant.
  // Average nearest neighbour distance is used as a proxy for a
  RangeConf<double> section_kx;
  RangeConf<double> section_ky;
  RangeConf<double> sdf_kx;
  RangeConf<double> sdf_ky;
  // Set Emin=Emax to automatically set E range
  RangeConf<double> sdf_e;
  // lattice point input (could possibly take special strings to do common
  // lattices)
  double fixed_e;
  std::string point_path;
  // Output file name
  std::string fname_h5;
  // Generally what I want. Not any more expensive than computing DOS and the
  // rest can be obtained by slicing this dataset.
  bool do_full_sdf;
  // Only set if you don't want to output a full sdf to disk. Will be ignored if
  // do_full_sdf is set.
  bool do_dos;
  bool do_e_section;
  bool do_path;
};

Delta delta(const VectorXd& d, RangeConf<f64> ec, f64 sharpening, f64 cutoff);

VectorXcd plane_wave(Vector2d k, const std::vector<Point>& points);
// VectorXcd plane_wave(Vector2d k, const std::vector<Pt2>& points);
MatrixXd disp(const VectorXd& d, const MatrixXcd& uh,
              const std::vector<Point>& points, f64 lat_const,
              std::vector<RangeConf<Vector2d>> kc, RangeConf<f64>& ec,
              f64 sharpening, f64 cutoff, bool print_progress = true);
std::vector<f64> dos(const VectorXd& d, const MatrixXcd& uh,
                     const std::vector<Point>& points, f64 lat_const,
                     RangeConf<f64> kxc, RangeConf<f64> kyc, RangeConf<f64>& ec,
                     f64 sharpening, f64 cutoff, bool print_progress = true);

// std::vector<f32> GPUEsection(Manager& m, const VectorXd& D, const MatrixXcd&
// UH,
//                              const std::vector<Point>& points, f64 lat_const,
//                              RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
//                              f64 sharpening, f64 cutoff);
MatrixXd e_section(const VectorXd& d, const MatrixXcd& uh,
                   const std::vector<Point>& points, f64 lat_const,
                   RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
                   f64 sharpening, f64 cutoff, bool print_progress = true);
std::vector<f64> full_sdf(const VectorXd& d, const MatrixXcd& uh,
                          const std::vector<Point>& points, f64 lat_const,
                          RangeConf<f64> kxc, RangeConf<f64> kyc,
                          RangeConf<f64>& ec, f64 sharpening, f64 cutoff,
                          bool print_progress = true);

template <class Func>
MatrixXd finite_hamiltonian(u32 n_points, const std::vector<Neighbour>& nbs,
                            Func f) {
  MatrixXd ham = MatrixXd::Zero(n_points, n_points);
  for (const auto& nb : nbs) {
    f64 val = f(nb.d);
    ham(static_cast<s64>(nb.i), static_cast<s64>(nb.j)) = val;
    ham(static_cast<s64>(nb.j), static_cast<s64>(nb.i)) = val;
  }
  return ham;
}
MatrixXd points_to_finite_hamiltonian(const std::vector<Point>& points,
                                      const kdt::KDTree<Point>& kdtree,
                                      f64 radius);

std::optional<SdfConf> toml_to_sdf_conf(const std::string& toml_path);
int do_sdf_calcs(SdfConf& conf);
