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

inline VectorXcd plane_wave(const MatrixXcd& restricted_uh, Vector2d k,
                            const Eigen::MatrixX2d& points) {
  return VectorXcd{
      (1. / std::sqrt(points.size())) *
      (restricted_uh * (points * (c64{0, 1} * k)).array().exp().matrix())};
}

struct NonHermDosConf : public SdfConf {
  RangeConf<f64> kxc;
  RangeConf<f64> kyc;
  RangeConf<f64> ec;
  f64 rscale;
  f64 lat_const;

  void run() const override {
    VectorXcd diag;
    MatrixXcd ph;
    HighFive::File pc{point_path, HighFive::File::ReadOnly};
    auto points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    if (saved_diag != "" && std::ifstream{saved_diag.c_str()}.good()) {
      HighFive::File diag_file{saved_diag, HighFive::File::ReadOnly};
      diag = diag_file.getDataSet("D").read<Eigen::VectorXcd>();
      ph = diag_file.getDataSet("P").read<Eigen::MatrixXcd>();
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
      ph = solver.eigenvectors();
      if (save_diag && saved_diag != "") {
        HighFive::File diag_file { saved_diag, }
      }
    }
    const u32 its = ec.n / 10;
    // MatrixXd sdf(kyc.n, kxc.n);
    VectorXd dos = VectorXd::Zero(static_cast<s64>(ec.n));
    // auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
    auto del = delta(diag, ec, sharpening, cutoff);
    if (print_progress) {
      std::cout << "[" << std::flush;
    }
#pragma omp parallel for
    for (u32 k = 0; k < ec.n; ++k) {

      auto coeffs_and_indices = del[k];
      const std::vector<size_t> indices = [&]() {
        std::vector<size_t> tmp;
        for (const auto pair : coeffs_and_indices) {
          tmp.push_back(pair.second);
        }
        return tmp;
      }();

      MatrixXcd restricted_uh = uh(indices, Eigen::indexing::all);

      for (u64 i = 0; i < kxc.n; i++) {

        const f64 kx = kxc.ith(i) * 2 * M_PI / lat_const;

        for (u64 j = 0; j < kyc.n; j++) {

          const f64 ky = kyc.ith(j) * 2 * M_PI / lat_const;
          const VectorXcd k_vec = plane_wave(restricted_uh, {kx, ky}, points);
          u32 cur_element = 0;

          for (const auto& pair : coeffs_and_indices) {

            dos(k) += pair.first * std::norm(k_vec(cur_element));
            cur_element += 1;
          }
        }
      }
      if (print_progress) {
        if (k % its == 0) {
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
