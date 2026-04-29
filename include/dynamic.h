#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "geometry.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include "io.h"
#include "spdlog/spdlog.h"
#include "typedefs.h"
#include <gsl/gsl_sf.h>
#include <random>

using Eigen::SparseMatrix, Eigen::VectorXcd, Eigen::MatrixXd;

static const Eigen::IOFormat oneliner(Eigen::StreamPrecision,
                                      Eigen::DontAlignCols, " ", " ", "", "",
                                      "", "");

typedef struct {
  f64 p;
  f64 alpha;
  f64 j;
  f64 rscale;
} Params;

constexpr HighFive::CompoundType compoundParams() {
  return {{"p", HighFive::create_datatype<f64>()},
          {"alpha", HighFive::create_datatype<f64>()},
          {"j", HighFive::create_datatype<f64>()},
          {"rscale", HighFive::create_datatype<f64>()}};
}

HIGHFIVE_REGISTER_TYPE(Params, compoundParams);

#define SET_STRUCT_FIELD(key, tbl)                                             \
  if (tbl.contains(#key))                                                      \
  key = *tbl[#key].value<decltype(key)>()

struct SimConf {
  virtual void run() const = 0;
  // {
  //   spdlog::warn("SimConf: run. Uh oh, this should never get called...");
  // }
  virtual ~SimConf() {}
};

template <class F, class State>
State rk4step(const State& x, f64 dt, F rhs) {
  spdlog::debug("Function: rk4step.");
  State k1 = rhs(x);
  State k2 = rhs(x + 0.5 * dt * k1);
  State k3 = rhs(x + 0.5 * dt * k2);
  State k4 = rhs(x + dt * k3);
  State ret = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
  spdlog::debug("Exiting rk4step.");
  return ret;
}

template <class Func>
struct RK4 {
  VectorXcd k1;
  VectorXcd k2;
  VectorXcd k3;
  VectorXcd k4;
  Func f;

  RK4(Func f, s64 n) : f{f}, k1{n}, k2{}, k3{}, k4{} {}

  VectorXcd step(VectorXcd& y, f64 dt) {

    k1 = f(y);
    k2 = f(y + 0.5 * dt * k1);
    k3 = f(y + 0.5 * dt * k2);
    k4 = f(y + 0.5 * dt * k3);
    y += (dt / 6.0) * ((k1 + k4) + 2 * (k2 + k3));
  }
};

inline auto kuramoto(f64 K, u32 N, const VectorXd& omega) {
  return [&, K, N](const VectorXd& theta) {
    MatrixXd sins = (VectorXd::Ones(N) * theta.transpose() -
                     theta * VectorXd::Ones(N).transpose())
                        .array()
                        .sin();
    return VectorXd(omega + (K / N) * (sins).rowwise().sum());
  };
}

struct KuramotoConf : public SimConf {
  std::string outfile;
  f64 K;
  u32 N;
  RangeConf<f64> t;

  KuramotoConf(const toml::table& tbl) {
    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(K, tbl);
    SET_STRUCT_FIELD(N, tbl);
    t = tblToRange(*tbl["t"].as_table());
  }

  void run() const override {
    VectorXd theta = VectorXd::LinSpaced(10, -1, 1);
    VectorXd omega = VectorXd::LinSpaced(10, 1, -1);
    /*std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(0, M_PI);
    for (auto& e : theta) {
      e = dis(gen);
    }
    for (auto& e : omega) {
      e = 0.01 * (dis(gen) - M_PI_2);
    }*/
    std::ofstream fout(outfile);
    fout << theta.format(oneliner) << '\n';
    auto rhs = kuramoto(K, N, omega);
    for (u32 i = 0; i < t.n; i++) {
      theta = rk4step(theta, t.d(), rhs);
      fout << theta.format(oneliner) << '\n';
    }

    fout.close();
  }
};

inline auto basic(const SparseMatrix<c64>& iH) {
  return [&](const VectorXcd& x) { return VectorXcd(iH * x); };
}

struct BasicConf {
  std::string outfile;
  std::string point_path;
  std::optional<f64> search_radius;
  RangeConf<f64> t;

  BasicConf(const toml::table& tbl) {

    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(point_path, tbl);
    if (tbl.contains("search_radius")) {
      search_radius = tbl["search_radius"].value<f64>().value();
    }
    t = tblToRange(*tbl["t"].as_table());
  }
};

struct BasicDistanceConf {
  std::string outfile;
  f64 alpha;
  f64 p;
  f64 j;
  RangeConf<f64> sep;
  RangeConf<f64> t;

  BasicDistanceConf(const toml::table& tbl) {
    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(alpha, tbl);
    SET_STRUCT_FIELD(p, tbl);
    SET_STRUCT_FIELD(j, tbl);
    t = tblToRange(*tbl["t"].as_table());
    sep = tblToRange(*tbl["sep"].as_table());
  }
};

struct BasicNLinConf {
  std::string outfile;
  std::string point_path;
  std::optional<f64> search_radius;
  f64 alpha;
  RangeConf<f64> t;

  BasicNLinConf(const toml::table& tbl) {

    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(point_path, tbl);
    SET_STRUCT_FIELD(alpha, tbl);
    if (tbl.contains("search_radius")) {
      search_radius = tbl["search_radius"].value<f64>().value();
    }
    t = tblToRange(*tbl["t"].as_table());
  }
};

inline auto basic_hankel(f64 p, f64 alpha, f64 j, f64 sep) {
  return [=](Eigen::Vector2cd psi) {
    return Eigen::Vector2cd(
        p * psi - c64{1, alpha} * psi.cwiseAbs2().cwiseProduct(psi) +
        j * c64{gsl_sf_bessel_J0(sep), gsl_sf_bessel_Y0(sep)} *
            psi(Eigen::indexing::lastN(2).reverse()));
  };
}

inline auto hankel_dd(const SparseMatrix<c64>& J, f64 p, f64 alpha) {
  return [&, p, alpha](Eigen::VectorXcd psi) {
    return p * psi - c64{1, alpha} * psi.cwiseAbs2().cwiseProduct(psi) +
           J * psi;
  };
}

struct HankelConf : public SimConf {
  std::string outfile;
  std::string point_path;
  std::optional<f64> search_radius;
  f64 p;
  f64 alpha;
  f64 j;
  f64 rscale;
  RangeConf<f64> t;

  HankelConf(const toml::table& tbl) {

    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(point_path, tbl);
    SET_STRUCT_FIELD(alpha, tbl);
    SET_STRUCT_FIELD(p, tbl);
    SET_STRUCT_FIELD(j, tbl);
    SET_STRUCT_FIELD(rscale, tbl);
    if (tbl.contains("search_radius")) {
      search_radius = tbl["search_radius"].value<f64>().value();
    }
    t = tblToRange(*tbl["t"].as_table());
  }

  void run() const override {
    spdlog::debug("Function: doBasicHankelDD.");
    const auto start = std::chrono::high_resolution_clock::now();
    HighFive::File pc(point_path, HighFive::File::ReadOnly);
    spdlog::debug("Read pointfile.");
    Eigen::MatrixX2d points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    spdlog::debug("Read points.");
    Eigen::MatrixX2i couplings =
        pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
    spdlog::debug("Read couplings.");
    const f64 r = rscale;
    const SparseMatrix<c64> J =
        j * SparseC(points, couplings, [r](Vector2d d) {
          spdlog::debug("Effective distance is: {}", r * d.norm());
          return c64{gsl_sf_bessel_J0(r * d.norm()),
                     gsl_sf_bessel_Y0(r * d.norm())};
        });
    spdlog::debug("Made sparse matrix J.");
    auto sol = Eigen::ComplexEigenSolver<MatrixXcd>(MatrixXcd(J));
    auto min_coeff = std::ranges::min_element(
        sol.eigenvalues(), [](c64 a, c64 b) { return a.imag() < b.imag(); });
    std::cout << "Highest eigenvalue is: " << min_coeff->imag() << '\n';
    const f64 eff_p = (p - 1) * min_coeff->imag();
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
    spdlog::debug("Allocating psi.");
    VectorXcd psi(points.rows());
    u64 n = psi.rows();
    u64 m = psi.cols();
    spdlog::debug("Allocated psi with dims {}x{}", n, m);
    spdlog::debug("Writing random coordinates to psi.");
    for (u64 i = 0; i < static_cast<u64>(psi.size()); i++) {
      auto x = dis(gen);
      psi[i] = {1e-4 * cos(x), 1e-4 * sin(x)};
    }
    const u32 overall_size = t.n * psi.size();
    spdlog::debug("Allocating psipdata with {} elements.", overall_size);
    MatrixXcd psidata(psi.size(), t.n + 1);
    const size_t byte_size = psi.size() * sizeof(c64);
    psidata(Eigen::indexing::all, 0) = psi;
    std::cout << "byte_size is: " << byte_size << '\n';
    std::cout << "Byte size of psidata is: " << psidata.size() * sizeof(c64);
    auto rhs = hankel_dd(J, eff_p, alpha);
    std::cout << t.n << '\n';
    for (u32 i = 1; i < t.n + 1; i++) {
      psi = rk4step(psi, t.d(), rhs);
      spdlog::debug("Return psiish with dims {}x{}", psi.rows(), psi.cols());
      spdlog::debug("Copying to psipdata.");
      psidata(Eigen::indexing::all, i) = psi;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::debug("Simulation took {} ms", duration.count());

    HighFive::File file(outfile, HighFive::File::Truncate);
    spdlog::debug("Writing psip to file.");

    file.createDataSet("psi", psidata);
    file.createDataSet("points", points);
    file.createDataSet("couplings", couplings);
    file.createDataSet("time", linspace(t, true));
    HighFive::DataSet paramSet =
        file.createDataSet("params", Params{p, alpha, j, rscale});
  }
  ~HankelConf() = default;
};

struct HankelTimeScanConf : public SimConf {
  std::string outfile;
  std::string point_path;
  RangeConf<f64> ps;
  RangeConf<f64> alphas;
  RangeConf<f64> js;
  RangeConf<f64> rscales;
  RangeConf<f64> t;

  HankelTimeScanConf(const toml::table& tbl) {
    spdlog::debug("Constructor: HankelScanConf(const toml::table& tbl)");
    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(point_path, tbl);
    ps = tblToRange(*tbl["ps"].as_table());
    alphas = tblToRange(*tbl["alphas"].as_table());
    js = tblToRange(*tbl["js"].as_table());
    rscales = tblToRange(*tbl["rscales"].as_table());
    t = tblToRange(*tbl["t"].as_table());
  }

  void run() const override {
    spdlog::debug("HankelTimeScanConf: method run.");
    const auto start = std::chrono::high_resolution_clock::now();
    HighFive::File pc(point_path, HighFive::File::ReadOnly);
    spdlog::debug("Read pointfile.");
    Eigen::MatrixX2d points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    spdlog::debug("Read points.");
    Eigen::MatrixX2i couplings =
        pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
    spdlog::debug("Read couplings.");

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(0.0, 2 * M_PI);

    s64 samples = ps.n * alphas.n * js.n * rscales.n;
    std::vector<c64> data(samples * 2 * (t.n + 1));
    s64 datasize = data.size();
    std::vector<c64> snapshotdata(samples * points.rows());
    spdlog::debug("data has {} elements in total.", datasize);
    s64 psize = ps.n;
    spdlog::debug("number of ps is {}.", psize);
    s64 alphasize = alphas.n;
    spdlog::debug("number of alphas is {}.", alphasize);
    s64 jsize = js.n;
    spdlog::debug("number of js is {}.", jsize);
    s64 rscalesize = rscales.n;
    spdlog::debug("number of rscales is {}.", rscalesize);
    VectorXcd init_psi(points.rows());
    for (u64 o = 0; o < (u64)init_psi.size(); ++o) {
      auto x = dis(gen);
      init_psi[o] = {1e-4 * cos(x), 1e-4 * sin(x)};
    }
    const size_t byte_size = init_psi.size() * sizeof(c64);
#pragma omp parallel for collapse(4)
    for (s64 m = 0; m < jsize; ++m) {
      const f64 j = js.ith(m);
      for (s64 n = 0; n < rscalesize; ++n) {
        const f64 scale = rscales.ith(n);
        const SparseMatrix<c64> J =
            j * SparseC(points, couplings, [scale](Vector2d d) {
              return c64{gsl_sf_bessel_J0(scale * d.norm()),
                         gsl_sf_bessel_Y0(scale * d.norm())};
            });
        const auto sol = Eigen::ComplexEigenSolver<MatrixXcd>(MatrixXcd(J));
        auto min_coeff =
            std::ranges::min_element(sol.eigenvalues(), [](c64 a, c64 b) {
              return a.imag() < b.imag();
            });
        for (s64 k = 0; k < psize; ++k) {
          const f64 p = ps.ith(k);
          for (s64 l = 0; l < alphasize; ++l) {
            const f64 alpha = alphas.ith(l);
            const f64 eff_p = (p - 1) * min_coeff->imag();
            spdlog::debug("Made sparse matrix J.");
            spdlog::debug("Allocating psi.");
            VectorXcd psi = init_psi;
            spdlog::debug("Writing random coordinates to psi.");
            // const size_t byte_size = psi.size() * sizeof(c64);
            auto rhs = hankel_dd(J, eff_p, alpha);
            spdlog::debug("Running rk4 for {} steps.", t.n);
            c64 psisum = psi.sum();
            s64 idx = 2 * (t.n + 1) *
                      (l + alphasize * (k + psize * (n + rscalesize * m)));
            data[idx] = psisum;
            data[idx + 1] = psi[0];
            idx += 2;
            for (u32 o = 1; o < t.n + 1; ++o) {
              psi = rk4step(psi, t.d(), rhs);
              c64 psisum = psi.sum();
              data[idx] = psisum;
              data[idx + 1] = psi[0];
              idx += 2;
            }
            s64 snapshot_idx =
                (n + rscalesize * (m + jsize * (l + alphasize * k))) *
                points.rows();
            memcpy(snapshotdata.data() + snapshot_idx, psi.data(), byte_size);
          }
        }
      }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end - start);
    spdlog::info("Simulation took {} s.", duration.count());

    HighFive::File file(outfile, HighFive::File::Truncate);
    spdlog::debug("Writing samples to file.");

    auto seriesSet = file.createDataSet<c64>(
        "sumpsitimeseries",
        HighFive::DataSpace(
            {static_cast<u64>(psize), static_cast<u64>(alphasize),
             static_cast<u64>(jsize), static_cast<u64>(rscalesize),
             static_cast<u64>(t.n + 1), 2}));
    seriesSet.write_raw(data.data());
    auto snapshot = file.createDataSet<c64>(
        "psisnapshot",
        HighFive::DataSpace(
            {static_cast<u64>(psize), static_cast<u64>(alphasize),
             static_cast<u64>(jsize), static_cast<u64>(rscalesize),
             static_cast<u64>(points.rows())}));
    snapshot.write_raw(snapshotdata.data());
    file.createDataSet("points", points);
    file.createDataSet("couplings", couplings);
    file.createDataSet("time", linspace(t, true));
    file.createDataSet("ps", linspace(ps, false));
    file.createDataSet("alphas", linspace(alphas, false));
    file.createDataSet("js", linspace(js, false));
    file.createDataSet("rscales", linspace(rscales, false));
    spdlog::debug("HankelTimeScanConf: exiting method run.");
  }

  ~HankelTimeScanConf() = default;
};

struct HankelScanConf : public SimConf {
  std::string outfile;
  std::string point_path;
  RangeConf<f64> ps;
  RangeConf<f64> alphas;
  RangeConf<f64> js;
  RangeConf<f64> rscales;
  RangeConf<f64> t;

  HankelScanConf(const toml::table& tbl) {
    spdlog::debug("Constructor: HankelScanConf(const toml::table& tbl)");
    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(point_path, tbl);
    ps = tblToRange(*tbl["ps"].as_table());
    alphas = tblToRange(*tbl["alphas"].as_table());
    js = tblToRange(*tbl["js"].as_table());
    rscales = tblToRange(*tbl["rscales"].as_table());
    t = tblToRange(*tbl["t"].as_table());
  }

  void run() const override {

    spdlog::debug("HankelScanConf: method run.");
    const auto start = std::chrono::high_resolution_clock::now();
    spdlog::info("");
    HighFive::File pc(point_path, HighFive::File::ReadOnly);
    spdlog::debug("Read pointfile.");
    Eigen::MatrixX2d points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    spdlog::debug("Read points.");
    Eigen::MatrixX2i couplings =
        pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
    spdlog::debug("Read couplings.");

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(0.0, 2 * M_PI);

    const s64 samples = ps.n * alphas.n * js.n * rscales.n;
    std::vector<c64> data(samples * points.rows());
    s64 datasize = data.size();
    spdlog::debug("data has {} elements in total.", datasize);
    s64 psize = ps.n;
    spdlog::debug("number of ps is {}.", psize);
    s64 alphasize = alphas.n;
    spdlog::debug("number of alphas is {}.", alphasize);
    s64 jsize = js.n;
    spdlog::debug("number of js is {}.", jsize);
    s64 rscalesize = rscales.n;
    spdlog::debug("number of rscales is {}.", rscalesize);
    VectorXcd init_psi(points.rows());
    for (u64 o = 0; o < static_cast<u64>(init_psi.size()); ++o) {
      auto x = dis(gen);
      *(init_psi.data() + o) = {1e-4 * cos(x), 1e-4 * sin(x)};
    }
    for (s64 m = 0; m < jsize; ++m) {
      const f64 j = js.ith(m);
#pragma omp parallel for
      for (s64 n = 0; n < rscalesize; ++n) {
        const f64 scale = rscales.ith(n);
        const SparseMatrix<c64> J =
            j * SparseC(points, couplings, [scale](Vector2d d) {
              return c64{gsl_sf_bessel_J0(scale * d.norm()),
                         gsl_sf_bessel_Y0(scale * d.norm())};
            });
        const auto sol = Eigen::ComplexEigenSolver<MatrixXcd>(MatrixXcd(J));
        auto min_coeff =
            std::ranges::min_element(sol.eigenvalues(), [](c64 a, c64 b) {
              return a.imag() < b.imag();
            });
        for (s64 k = 0; k < psize; ++k) {
          const f64 p = ps.ith(k);
          for (s64 l = 0; l < alphasize; ++l) {
            const f64 alpha = alphas.ith(l);
            const f64 eff_p = (p - 1) * min_coeff->imag();
            spdlog::debug("Made sparse matrix J.");
            spdlog::debug("Allocating psi.");
            VectorXcd psi = init_psi;
            spdlog::debug("Writing random coordinates to psi.");
            const size_t byte_size = psi.size() * sizeof(c64);
            auto rhs = hankel_dd(J, eff_p, alpha);
            spdlog::debug("Running rk4 for {} steps.", t.n);
            for (u32 o = 1; o < t.n + 1; ++o) {
              psi = rk4step(psi, t.d(), rhs);
            }
            s64 idx = (n + rscalesize * (m + jsize * (l + alphasize * k))) *
                      points.rows();
            spdlog::debug("Copying to {}-th complex number.", idx);
            spdlog::debug("Finished calculating.");
            memcpy(data.data() + idx, psi.data(), byte_size);
          }
        }
      }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end - start);
    spdlog::info("Simulation took {} seconds", duration.count());

    HighFive::File file(outfile, HighFive::File::Truncate);
    spdlog::debug("Writing samples to file.");

    auto sampleSet = file.createDataSet<c64>(
        "psis", HighFive::DataSpace(
                    {static_cast<u64>(psize), static_cast<u64>(alphasize),
                     static_cast<u64>(jsize), static_cast<u64>(rscalesize),
                     static_cast<u64>(points.rows())}));
    sampleSet.write_raw(data.data());
    file.createDataSet("points", points);
    file.createDataSet("couplings", couplings);
    file.createDataSet("time", linspace(t, true));
    file.createDataSet("ps", linspace(ps, false));
    file.createDataSet("alphas", linspace(alphas, false));
    file.createDataSet("js", linspace(js, false));
    file.createDataSet("rscales", linspace(rscales, false));
    spdlog::debug("HankelScanConf: exiting method run.");
  }

  ~HankelScanConf() = default;
};
#undef SET_STRUCT_FIELD

struct DelayConf {
  std::string outfile;
  std::string point_path;
  std::optional<f64> search_radius;
  f64 p;
  f64 alpha;
  f64 j;
  f64 v;
  RangeConf<f64> t;
};

// struct DynConf {
//   std::optional<KuramotoConf> kuramoto;
//   std::optional<BasicConf> basic;
//   std::optional<BasicDistanceConf> bd;
//   std::optional<BasicNLinConf> basicnlin;
//   std::vector<TETMConf> tetm;
//   std::optional<HankelScanConf> hsc;
//   std::vector<HankelScanConf> hscs;
// };

enum Sims {
  eHankelTimeScan,
  eHankelScan,
  eHankel,
};

static const std::unordered_map<std::string_view, Sims> sim_types{
    {"hankelscan", Sims::eHankelScan},
    {"hankeltimescan", Sims::eHankelTimeScan},
    {"hankel", eHankel},
};

inline std::vector<std::unique_ptr<SimConf>>
toml_to_dyn_conf(const std::string& fname) {
  toml::table tbl;
  spdlog::debug("Function: toml_to_dyn_conf");
  try {
    tbl = toml::parse_file(fname);
  } catch (const std::exception& err) {
    std::cerr << "Parsing file " << fname
              << " failed with exception: " << err.what() << '\n';
    return {};
  }
  spdlog::debug("File {} successfully parsed", fname);
  std::vector<std::unique_ptr<SimConf>> retVec;
  tbl.for_each([&](const toml::key& key, toml::array& val) {
    spdlog::debug("Found key: {}", key.str());
    switch (sim_types.at(key.str())) {
    case eHankelTimeScan:
      val.for_each([&](toml::table& arrtbl) {
        spdlog::debug("toml_to_dyn_conf: Pushing back hankel time scan conf.");
        HankelTimeScanConf tmp(arrtbl);
        retVec.push_back(
            std::unique_ptr<SimConf>(new HankelTimeScanConf(arrtbl)));
      });
      break;
    case eHankelScan:
      val.for_each([&](toml::table& arrtbl) {
        spdlog::debug("toml_to_dyn_conf: Pushing back hankel scan conf.");
        retVec.push_back(std::unique_ptr<SimConf>(new HankelScanConf(arrtbl)));
      });
      break;
    case eHankel:
      val.for_each([&](toml::table& arrtbl) {
        spdlog::debug("toml_to_dyn_conf: Pushing back hankel conf.");
        retVec.push_back(std::unique_ptr<SimConf>(new HankelConf(arrtbl)));
      });
      break;
    }
  });
  return retVec;
}

// auto basic(const SparseMatrix<c64>& iH);
// int doBasic(const BasicConf& conf);
// int doBasicNLin(const BasicNLinConf& conf);
// int doExactBasic(const BasicConf& conf);
// int doKuramoto(const KuramotoConf& conf);
// int doTETM(const TETMConf& conf);
// int doDistanceScan(const BasicDistanceConf& conf);
// int doNoCoupling(const BasicDistanceConf& conf);
// int doNCDD(const BasicDistanceConf& conf);
// int doBasicHankelDD(const std::vector<TETMConf>& conf);
// int doHankelScan(const std::vector<HankelScanConf>& conf);
// int doHankelTimeScan(const std::vector<HankelScanConf>& conf);
// int GPUHankelTimeScan(const std::vector<HankelScanConf>& conf);
