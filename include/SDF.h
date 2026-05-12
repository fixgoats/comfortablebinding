#pragma once
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "geometry.h"
#include "gsl/gsl_sf.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include "io.h"
#include "localtomlpp.h"
#include "mathhelpers.h"
#include <fstream>
#include <iostream>

#define DEBUG_START spdlog::debug("{}: Start", __func__)
#define DEBUG_END spdlog::debug("{}: End", __func__)
#define DEBUG_LOG(...)                                                         \
  spdlog::debug("{}: {}", __func__, std::format(__VA_ARGS__))

using Eigen::VectorXcd, Eigen::VectorXd, Eigen::Vector2d, Eigen::MatrixXcd,
    Eigen::MatrixXd, Eigen::SparseMatrix, Eigen::Vector2i;

typedef std::vector<std::vector<std::pair<f64, u32>>> Delta;

Delta delta(const VectorXd& d, RangeConf<f64> ec, f64 sharpening, f64 cutoff);
Delta delta(const VectorXcd& d, RangeConf<f64> ec, f64 sharpening, f64 cutoff);
inline VectorXcd plane_wave(Vector2d k, const Eigen::MatrixX2d& points) {
  VectorXcd v = c64{0, -1} * points * k;
  return VectorXcd{v.array().exp()};
}

inline VectorXcd plane_wave(const MatrixXcd& proj, Vector2d k,
                            const Eigen::MatrixX2d& points) {
  auto vec = (1. / std::sqrt(points.size())) *
             (points * (c64{0, 1} * k)).array().exp().matrix();
  return VectorXcd{proj.colPivHouseholderQr().solve(vec)};
}

enum SdfSims : u8 {
  eNonHermDos,
  eHermDos,
  eSimplerHermDos,
  eHermDisp,
};

#define SET_STRUCT_FIELD(key, tbl)                                             \
  if ((tbl).contains(#key))                                                    \
  (key) = *(tbl)[#key].value<decltype(key)>()

struct SdfConf {
  f64 sharpening;
  f64 cutoff;
  std::string point_path;
  std::string out_path;
  std::string saved_diag;
  bool save_diag;
  bool print_progress = true;

  SdfConf(const toml::table& tbl) {
    DEBUG_START;
    DEBUG_LOG("Initializing sharpening");
    SET_STRUCT_FIELD(sharpening, tbl);
    DEBUG_LOG("Initializing cutoff");
    SET_STRUCT_FIELD(cutoff, tbl);
    DEBUG_LOG("Initializing point_path");
    SET_STRUCT_FIELD(point_path, tbl);
    DEBUG_LOG("Initializing out_path");
    SET_STRUCT_FIELD(out_path, tbl);
    DEBUG_LOG("Initializing saved_diag");
    SET_STRUCT_FIELD(saved_diag, tbl);
    DEBUG_LOG("Initializing save_diag");
    SET_STRUCT_FIELD(save_diag, tbl);
    DEBUG_LOG("Initializing print_progress");
    SET_STRUCT_FIELD(print_progress, tbl);
    DEBUG_END;
  }
  virtual void run() const = 0;
  virtual ~SdfConf() = default;
};

struct NonHermDosConf : public SdfConf {
  RangeConf<f64> kxc;
  RangeConf<f64> kyc;
  RangeConf<f64> ec;
  f64 rscale;

  NonHermDosConf(const toml::table& tbl) : SdfConf(tbl) {
    DEBUG_START;
    DEBUG_LOG("Initializing kxc");
    kxc = tblToRange(*tbl["kxc"].as_table());
    DEBUG_LOG("Initializing kyc");
    kyc = tblToRange(*tbl["kyc"].as_table());
    DEBUG_LOG("Initializing ec");
    ec = tblToRange(*tbl["ec"].as_table());
    DEBUG_LOG("Initializing rscale");
    SET_STRUCT_FIELD(rscale, tbl);
  }

  void run() const override {
    VectorXcd diag;
    MatrixXcd proj;
    HighFive::File pc{point_path, HighFive::File::ReadOnly};
    auto points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    if (saved_diag != "" && std::ifstream{saved_diag.c_str()}.good()) {
      HighFive::File diag_file{saved_diag, HighFive::File::ReadOnly};
      // auto new_diag = ;
      diag = diag_file.getDataSet("D").read<Eigen::VectorXcd>(); // new_diag;
      // auto new_proj = ;
      proj = diag_file.getDataSet("P").read<Eigen::MatrixXcd>(); // new_proj;
    } else {
      MatrixXcd coupling_mat;
      auto couplings = pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
      f64 r = rscale;
      coupling_mat = sparse_c(points, couplings, [r](Vector2d d) {
        return c64{gsl_sf_bessel_J0(r * d.norm()),
                   gsl_sf_bessel_Y0(r * d.norm())};
      });
      Eigen::ComplexEigenSolver<MatrixXcd> solver{coupling_mat};
      diag = solver.eigenvalues();
      proj = solver.eigenvectors();
      if (save_diag && saved_diag != "") {
        HighFive::File diag_file{saved_diag, HighFive::File::Truncate};
        diag_file.createDataSet("D", diag);
        diag_file.createDataSet("P", proj);
      }
    }
    const u32 its = kxc.n / 10;
    // MatrixXd sdf(kyc.n, kxc.n);
    VectorXd dos = VectorXd::Zero(static_cast<s64>(ec.n));
    // auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
    auto del = delta(diag, ec, sharpening, cutoff);

    f64 cond = proj.bdcSvd().singularValues()(0) /
               proj.bdcSvd().singularValues()(
                   proj.bdcSvd().singularValues().size() - 1);
    std::cout << "Condition number: " << cond << '\n';
    MatrixXcd proj_inv = proj.inverse();
    if (print_progress) {
      std::cout << "[" << std::flush;
    }
    for (u64 i = 0; i < kxc.n; i++) {

      const f64 kx = kxc.ith(i);

      for (u64 j = 0; j < kyc.n; j++) {

        const f64 ky = kyc.ith(j);
        const VectorXcd k_vec = proj_inv * plane_wave({kx, ky}, points);
        for (u32 k = 0; k < ec.n; ++k) {

          auto coeffs_and_indices = del[k];

          for (const auto& pair : coeffs_and_indices) {
            dos(k) += pair.first * std::norm(k_vec(pair.second));
          }
        }
      }
      if (print_progress) {
        if (i % its == 0) {
          std::cout << "█|" << std::flush;
        }
      }
    }
    if (print_progress) {
      std::cout << "█]\n";
    }
    HighFive::File sdf_file(out_path, HighFive::File::Truncate);
    sdf_file.createDataSet("dos", dos);
  }
};

inline MatrixXd simple_h(const Eigen::MatrixX2d& points,
                         const Eigen::MatrixX2i& couplings) {
  MatrixXd ham(points.rows(), points.rows());
  for (s64 i = 0; i < couplings.rows(); ++i) {
    s64 j = couplings(i, 0);
    s64 k = couplings(i, 1);
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    ham(j, k) = 1;
    ham(k, j) = 1;
  }
  return ham;
}

struct HermDosConf : public SdfConf {
  RangeConf<f64> kxc;
  RangeConf<f64> kyc;
  RangeConf<f64> ec;

  HermDosConf(const toml::table& tbl) : SdfConf(tbl) {
    DEBUG_START;
    DEBUG_LOG("Initializing kxc");
    kxc = tblToRange(*tbl["kxc"].as_table());
    DEBUG_LOG("Initializing kyc");
    kyc = tblToRange(*tbl["kyc"].as_table());
    DEBUG_LOG("Initializing ec");
    ec = tblToRange(*tbl["ec"].as_table());
    DEBUG_END;
  }

  void run() const override {
    VectorXd diag;
    MatrixXcd proj;
    HighFive::File pc{point_path, HighFive::File::ReadOnly};
    auto points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    if (saved_diag != "" && std::ifstream{saved_diag.c_str()}.good()) {
      HighFive::File diag_file{saved_diag, HighFive::File::ReadOnly};
      diag = diag_file.getDataSet("D").read<Eigen::VectorXd>();
      proj = diag_file.getDataSet("P").read<Eigen::MatrixXcd>();
    } else {
      MatrixXcd coupling_mat;
      auto couplings = pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
      coupling_mat = simple_h(points, couplings);

      auto sol = hermitianEigenSolver(coupling_mat);
      // Eigen::HermitianEigenSolver<MatrixXd> solver{coupling_mat};
      diag = sol.D;
      proj = sol.U;
      if (save_diag && saved_diag != "") {
        HighFive::File diag_file{saved_diag, HighFive::File::Truncate};
        diag_file.createDataSet("D", diag);
        diag_file.createDataSet("P", proj);
      }
    }
    const u32 its = kxc.n / 10;
    // MatrixXd sdf(kyc.n, kxc.n);
    VectorXd dos = VectorXd::Zero(static_cast<s64>(ec.n));
    // auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
    auto del = delta(diag, ec, sharpening, cutoff);
    if (print_progress) {
      std::cout << "[" << std::flush;
    }
    for (u64 i = 0; i < kxc.n; i++) {

      const f64 kx = kxc.ith(i);

      for (u64 j = 0; j < kyc.n; j++) {

        const f64 ky = kyc.ith(j);
        const VectorXcd k_vec = plane_wave(proj, {kx, ky}, points);
        for (u32 k = 0; k < ec.n; ++k) {

          auto coeffs_and_indices = del[k];

          for (const auto& pair : coeffs_and_indices) {
            dos(k) += pair.first * std::norm(k_vec(pair.second));
          }
        }
      }
      if (print_progress) {
        if (i % its == 0) {
          std::cout << "█|" << std::flush;
        }
      }
    }
    if (print_progress) {
      std::cout << "█]\n";
    }
    HighFive::File sdf_file(out_path, HighFive::File::Truncate);
    sdf_file.createDataSet("dos", dos);
  }
};

struct SimplerHermDosConf : public SdfConf {
  RangeConf<f64> kxc;
  RangeConf<f64> kyc;
  RangeConf<f64> ec;

  SimplerHermDosConf(const toml::table& tbl) : SdfConf(tbl) {
    DEBUG_START;
    DEBUG_LOG("Initializing kxc");
    kxc = tblToRange(*tbl["kxc"].as_table());
    DEBUG_LOG("Initializing kyc");
    kyc = tblToRange(*tbl["kyc"].as_table());
    DEBUG_LOG("Initializing ec");
    ec = tblToRange(*tbl["ec"].as_table());
    DEBUG_END;
  }

  void run() const override {
    VectorXd diag;
    MatrixXcd proj;
    HighFive::File pc{point_path, HighFive::File::ReadOnly};
    auto points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    if (saved_diag != "" && std::ifstream{saved_diag.c_str()}.good()) {
      HighFive::File diag_file{saved_diag, HighFive::File::ReadOnly};
      diag = diag_file.getDataSet("D").read<Eigen::VectorXd>();
      proj = diag_file.getDataSet("P").read<Eigen::MatrixXcd>();
    } else {
      MatrixXcd coupling_mat;
      auto couplings = pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
      coupling_mat = simple_h(points, couplings);

      auto sol = hermitianEigenSolver(coupling_mat);
      // Eigen::HermitianEigenSolver<MatrixXd> solver{coupling_mat};
      diag = sol.D;
      proj = sol.U;
      if (save_diag && saved_diag != "") {
        HighFive::File diag_file{saved_diag, HighFive::File::Truncate};
        diag_file.createDataSet("D", diag);
        diag_file.createDataSet("P", proj);
      }
    }
    const u32 its = kxc.n / 10;
    // MatrixXd sdf(kyc.n, kxc.n);
    VectorXd dos = VectorXd::Zero(static_cast<s64>(ec.n));
    // auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
    auto del = delta(diag, ec, sharpening, cutoff);
    if (print_progress) {
      std::cout << "[" << std::flush;
    }
    MatrixXcd uh = proj.adjoint();
    for (u64 i = 0; i < kxc.n; i++) {

      const f64 kx = kxc.ith(i);

      for (u64 j = 0; j < kyc.n; j++) {

        const f64 ky = kyc.ith(j);
        const VectorXcd k_vec = uh * plane_wave({kx, ky}, points);
        for (u32 k = 0; k < ec.n; ++k) {

          auto coeffs_and_indices = del[k];

          for (const auto& pair : coeffs_and_indices) {
            dos(k) += pair.first * std::norm(k_vec(pair.second));
          }
        }
      }
      if (print_progress) {
        if (i % its == 0) {
          std::cout << "█|" << std::flush;
        }
      }
    }
    if (print_progress) {
      std::cout << "█]\n";
    }
    HighFive::File sdf_file(out_path, HighFive::File::Truncate);
    sdf_file.createDataSet("dos", dos);
  }
};

MatrixXd disp(f64 kx, RangeConf<f64> kyc, const Delta& del,
              const MatrixXcd& proj_inv, const Eigen::MatrixX2d& points) {
  auto sdf =
      MatrixXd::Zero(static_cast<s64>(del.size()), static_cast<s64>(kyc.n));
#pragma omp parallel for
  for (u32 i = 0; i < kyc.n; i++) {

    const f64 ky = kyc.ith(i);

    const VectorXcd k_vec = proj_inv * plane_wave({kx, ky}, points);
    for (u32 k = 0; k < del.size(); ++k) {

      auto coeffs_and_indices = del[k];

      for (const auto& pair : coeffs_and_indices) {
        sdf(k, i) += pair.first * std::norm(k_vec(pair.second));
      }
    }
    if (print_progress) {
      if (i % its == 0) {
        std::cout << "█|" << std::flush;
      }
    }
  }
}
struct HermDispConf : public SdfConf {
  RangeConf<f64> kxc;
  RangeConf<f64> ec;

  HermDispConf(const toml::table& tbl) : SdfConf(tbl) {
    DEBUG_START;
    DEBUG_LOG("Initializing kxc");
    kxc = tblToRange(*tbl["kxc"].as_table());
    DEBUG_LOG("Initializing ec");
    ec = tblToRange(*tbl["ec"].as_table());
    DEBUG_END;
  }

  void run() const override {
    VectorXd diag;
    MatrixXcd proj;
    HighFive::File pc{point_path, HighFive::File::ReadOnly};
    auto points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    if (saved_diag != "" && std::ifstream{saved_diag.c_str()}.good()) {
      HighFive::File diag_file{saved_diag, HighFive::File::ReadOnly};
      diag = diag_file.getDataSet("D").read<Eigen::VectorXd>();
      proj = diag_file.getDataSet("P").read<Eigen::MatrixXcd>();
    } else {
      MatrixXcd coupling_mat;
      auto couplings = pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
      coupling_mat = simple_h(points, couplings);

      auto sol = hermitianEigenSolver(coupling_mat);
      // Eigen::HermitianEigenSolver<MatrixXd> solver{coupling_mat};
      diag = sol.D;
      proj = sol.U;
      if (save_diag && saved_diag != "") {
        HighFive::File diag_file{saved_diag, HighFive::File::Truncate};
        diag_file.createDataSet("D", diag);
        diag_file.createDataSet("P", proj);
      }
    }
    const u32 its = kxc.n / 10;
    // MatrixXd sdf(kyc.n, kxc.n);
    // auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
    auto del = delta(diag, ec, sharpening, cutoff);
    if (print_progress) {
      std::cout << "[" << std::flush;
    }
    MatrixXcd uh = proj.adjoint();
    MatrixXd sdf =
        MatrixXd::Zero(static_cast<s64>(ec.n), static_cast<s64>(kxc.n));
#pragma omp parallel for
    for (u32 i = 0; i < kxc.n; i++) {

      const f64 kx = kxc.ith(i);

      const VectorXcd k_vec = uh * plane_wave({kx, 0}, points);
      for (u32 k = 0; k < ec.n; ++k) {

        auto coeffs_and_indices = del[k];

        for (const auto& pair : coeffs_and_indices) {
          sdf(k, i) += pair.first * std::norm(k_vec(pair.second));
        }
      }
      if (print_progress) {
        if (i % its == 0) {
          std::cout << "█|" << std::flush;
        }
      }
    }
    if (print_progress) {
      std::cout << "█]\n";
    }
    HighFive::File sdf_file(out_path, HighFive::File::Truncate);
    sdf_file.createDataSet("sdf", sdf);
  }
};

static const std::unordered_map<std::string_view, SdfSims> SDF_SIM_TYPES{
    {"non_herm_dos", SdfSims::eNonHermDos},
    {"herm_dos", SdfSims::eHermDos},
    {"simpler_herm_dos", SdfSims::eSimplerHermDos},
    {"herm_disp", SdfSims::eHermDisp},
};

inline std::vector<std::unique_ptr<SdfConf>>
toml_to_sdf_conf(const std::string& fname) {
  DEBUG_START;
  toml::table tbl;
  try {
    tbl = toml::parse_file(fname);
  } catch (const std::exception& err) {
    std::cerr << "Parsing file " << fname
              << " failed with exception: " << err.what() << '\n';
    return {};
  }
  DEBUG_LOG("File {} successfully parsed", fname);
  std::vector<std::unique_ptr<SdfConf>> ret_vec;
  tbl.for_each([&](const toml::key& key, toml::array& val) {
    spdlog::debug("toml_to_sdf_conf: Found key: {}", key.str());
    switch (SDF_SIM_TYPES.at(key.str())) {
    case eNonHermDos: {
      val.for_each([&](toml::table& arrtbl) {
        spdlog::debug("toml_to_sdf_conf: Pushing back dos conf.");
        ret_vec.push_back(std::unique_ptr<SdfConf>(new NonHermDosConf(arrtbl)));
      });
      break;
    }
    case eHermDos: {
      val.for_each([&](toml::table& arrtbl) {
        spdlog::debug("toml_to_sdf_conf: Pushing back herm dos conf.");
        ret_vec.push_back(std::unique_ptr<SdfConf>(new HermDosConf(arrtbl)));
      });
      break;
    }
    case eSimplerHermDos: {
      val.for_each([&](toml::table& arrtbl) {
        spdlog::debug("toml_to_sdf_conf: Pushing back simpler herm dos conf.");
        ret_vec.push_back(
            std::unique_ptr<SdfConf>(new SimplerHermDosConf(arrtbl)));
      });
      break;
    }
    case eHermDisp: {
      val.for_each([&](toml::table& arrtbl) {
        spdlog::debug("toml_to_sdf_conf: Pushing back disp conf.");
        ret_vec.push_back(std::unique_ptr<SdfConf>(new HermDispConf(arrtbl)));
      });
      break;
    }
    }
  });
  return ret_vec;
}

// MatrixXd e_section(const VectorXd& d, const MatrixXcd& uh,
//                    const std::vector<Point>& points, f64 lat_const,
//                    RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
//                    f64 sharpening, f64 cutoff, bool print_progress = true);
//
// std::vector<f64> full_sdf(const VectorXd& d, const MatrixXcd& uh,
//                           const std::vector<Point>& points, f64 lat_const,
//                           RangeConf<f64> kxc, RangeConf<f64> kyc,
//                           RangeConf<f64>& ec, f64 sharpening, f64 cutoff,
//                           bool print_progress = true);

// template <class Func>
// MatrixXd finite_hamiltonian(u32 n_points, const std::vector<Neighbour>& nbs,
//                             Func f) {
//   MatrixXd ham = MatrixXd::Zero(n_points, n_points);
//   for (const auto& nb : nbs) {
//     f64 val = f(nb.d);
//     ham(static_cast<s64>(nb.i), static_cast<s64>(nb.j)) = val;
//     ham(static_cast<s64>(nb.j), static_cast<s64>(nb.i)) = val;
//   }
//   return ham;
// }
// MatrixXd points_to_finite_hamiltonian(const std::vector<Point>& points,
//                                       const kdt::KDTree<Point>& kdtree,
//                                       f64 radius);

// std::optional<SdfConf> toml_to_sdf_conf(const std::string& toml_path);
// int do_sdf_calcs(SdfConf& conf);
#undef SET_STRUCT_FIELD
#undef DEBUG_END
#undef DEBUG_START
#undef DEBUG_LOG
