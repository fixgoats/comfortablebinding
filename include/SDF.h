#pragma once
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "geometry.h"
#include "gsl/gsl_sf.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include "mathhelpers.h"
#include <fstream>
#include <iostream>
// #include "vkcore.h"

using Eigen::VectorXcd, Eigen::VectorXd, Eigen::Vector2d, Eigen::MatrixXcd,
    Eigen::MatrixXd, Eigen::SparseMatrix, Eigen::Vector2i;

typedef std::vector<std::vector<std::pair<f64, u32>>> Delta;

Delta delta(const VectorXd& d, RangeConf<f64> ec, f64 sharpening, f64 cutoff);
Delta delta(const VectorXcd& d, RangeConf<f64> ec, f64 sharpening, f64 cutoff);
inline VectorXcd plane_wave(Vector2d k, const Eigen::MatrixX2d& points) {
  VectorXcd v = c64{0, -1} * points * k;
  return VectorXcd{v.array().exp()};
}

struct SdfConf {
  f64 sharpening;
  f64 cutoff;
  std::string point_path;
  std::string out_path;
  std::string saved_diag;
  bool save_diag;
  bool print_progress = true;

  virtual void run() const = 0;
  virtual ~SdfConf() = default;
};

struct NonHermDosConf : public SdfConf {
  RangeConf<f64> kxc;
  RangeConf<f64> kyc;
  RangeConf<f64> ec;
  f64 rscale;
  f64 lat_const;

  void run() const override {
    VectorXcd diag;
    MatrixXcd uh;
    HighFive::File pc{point_path, HighFive::File::ReadOnly};
    auto points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    if (saved_diag != "" && std::ifstream{saved_diag.c_str()}.good()) {
      HighFive::File diag_file{saved_diag, HighFive::File::ReadOnly};
      diag = diag_file.getDataSet("D").read<Eigen::VectorXcd>();
      uh = diag_file.getDataSet("U").read<Eigen::MatrixXcd>();
    } else {
      MatrixXcd coupling_mat;
      auto couplings = pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
      f64 r = rscale;
      coupling_mat = sparse_c(points, couplings, [r](Vector2d d) {
        return c64{gsl_sf_bessel_J0(r * d.norm()),
                   gsl_sf_bessel_Y0(r * d.norm())};
      });
      Eigen::EigenSolver<MatrixXcd> solver{coupling_mat};
      diag = solver.eigenvalues();
      uh = solver.eigenvectors();
      if (save_diag) {
      }
    }
    const u32 its = kxc.n / 10;
    MatrixXd sdf(kyc.n, kxc.n);
    VectorXd dos
        // auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
        auto del = delta(diag, ec, sharpening, cutoff);
    if (print_progress) {
      std::cout << "[" << std::flush;
    }
    const std::vector<size_t> indices = [&]() {
      std::vector<size_t> tmp;
      for (const auto pair : del[0]) {
        tmp.push_back(pair.second);
      }
      return tmp;
    }();
    MatrixXcd restricted_uh = uh(indices, Eigen::indexing::all);
#pragma omp parallel for
    for (size_t i = 0; i < kxc.n; i++) {
      const f64 kx = kxc.ith(i) * 2 * M_PI / lat_const;
      std::vector<u64> cur_element(kyc.n, 0);
      for (u64 j = 0; j < kyc.n; j++) {
        const f64 ky = kyc.ith(j) * 2 * M_PI / lat_const;
        const VectorXcd k_vec = restricted_uh * plane_wave({kx, ky}, points);
        u32 cur_element = 0;
        for (const auto& pair : del[0]) {
          sdf(static_cast<s64>(j), static_cast<s64>(i)) +=
              pair.first * std::norm(k_vec(cur_element));
          cur_element += 1;
        }
      }
      if (print_progress) {
        if (i % its == 0) {
          std::cout << "█|" << std::flush;
        }
      }
    }
#pragma omp barrier
    if (print_progress) {
      std::cout << "█]\n";
    }
    HighFive::File sdf_file(out_path, HighFive::File::ReadWrite);
    sdf_file.createDataSet("dos", dos);
  }
};

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
  // in units of "Brillouin zone", i.e. 1 = 2pi/a where a is a lattice
  // constant. Average nearest neighbour distance is used as a proxy for a
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
  // Only set if you don't want to output a full sdf to disk. Will be ignored
  // if do_full_sdf is set.
  bool do_dos;
  bool do_e_section;
  bool do_path;
};

VectorXcd plane_wave(Vector2d k, const std::vector<Point>& points);
// VectorXcd plane_wave(Vector2d k, const std::vector<Pt2>& points);
MatrixXd disp(const VectorXd& d, const MatrixXcd& uh,
              const std::vector<Point>& points, f64 lat_const,
              std::vector<RangeConf<Vector2d>> kc, RangeConf<f64>& ec,
              f64 sharpening, f64 cutoff, bool print_progress = true);
MatrixXd non_herm_disp(const VectorXcd& d, const MatrixXcd& uh,
                       const std::vector<Point>& points, f64 lat_const,
                       RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
                       f64 sharpening, f64 cutoff, bool print_progress);
std::vector<f64> dos(const VectorXd& d, const MatrixXcd& uh,
                     const std::vector<Point>& points, f64 lat_const,
                     RangeConf<f64> kxc, RangeConf<f64> kyc, RangeConf<f64>& ec,
                     f64 sharpening, f64 cutoff, bool print_progress = true);
MatrixXd non_herm_e_section(const VectorXd& d, const MatrixXcd& uh,
                            const std::vector<Point>& points, f64 lat_const,
                            RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
                            f64 sharpening, f64 cutoff, bool print_progress);
MatrixXd non_herm_dos(const VectorXd& d, const MatrixXcd& uh,
                      const std::vector<Point>& points, f64 lat_const,
                      RangeConf<f64> kxc, RangeConf<f64> kyc, RangeConf<f64> e,
                      f64 sharpening, f64 cutoff, bool print_progress);

// std::vector<f32> GPUEsection(Manager& m, const VectorXd& D, const
// MatrixXcd& UH,
//                              const std::vector<Point>& points, f64
//                              lat_const, RangeConf<f64> kxc, RangeConf<f64>
//                              kyc, f64 e, f64 sharpening, f64 cutoff);
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
