#include "dynamic.h"
#include "Eigen/Core"
#include "SDF.h"
#include "geometry.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include "io.h"
#include "spdlog/spdlog.h"
#include "unsupported/Eigen/MatrixFunctions"
#include "vkcore.hpp"
#include "vulkan/vulkan.hpp"
#include <gsl/gsl_sf.h>
#include <print>
#include <random>
#include <toml++/toml.hpp>
#include <vector>

using Eigen::MatrixXd, Eigen::VectorXcd, Eigen::MatrixX2cd, Eigen::VectorXcf,
    Eigen::MatrixXcf;

#define SET_STRUCT_FIELD(c, tbl, key)                                          \
  if (tbl.contains(#key))                                                      \
  c.key = *tbl[#key].value<decltype(c.key)>()

std::optional<DynConf> tomlToDynConf(const std::string& fname) {
  toml::table tbl;
  spdlog::debug("Function: tomlToDynConf");
  try {
    tbl = toml::parse_file(fname);
  } catch (const std::exception& err) {
    std::cerr << "Parsing file " << fname
              << " failed with exception: " << err.what() << '\n';
    return {};
  }
  spdlog::debug("File {} successfully parsed", fname);
  DynConf conf{};
  if (tbl.contains("basic")) {
    spdlog::debug("Writing config for basic.");
    conf.basic = BasicConf(*tbl["basic"].as_table());
  }
  if (tbl.contains("basicdistance")) {
    spdlog::debug("Writing config for distance scan");
    conf.bd = BasicDistanceConf(*tbl["basicdistance"].as_table());
  }
  if (tbl.contains("tetm")) {
    conf.tetm = TETMConf(*tbl["tetm"].as_table());
  }
  if (tbl.contains("hankelscan")) {
    conf.hsc = HankelScanConf(*tbl["hankelscan"].as_table());
  }
  if (tbl.contains("hankelscans")) {
    toml::array htbls = *tbl["hankelscans"].as_array();
    spdlog::debug("number of Hankel scan configs: {}", htbls.size());

    for (const auto& eee : htbls) {
      conf.hscs.emplace_back(*eee.as_table());
    }
  }
  if (tbl.contains("hankeltimescans")) {
    toml::array htbls = *tbl["hankeltimescans"].as_array();
    spdlog::debug("number of Hankel scan configs: {}", htbls.size());

    for (const auto& eee : htbls) {
      conf.hscs.emplace_back(*eee.as_table());
    }
  }
  if (tbl.contains("delay")) {
    conf.tetm = TETMConf(*tbl["delay"].as_table());
  }
  if (tbl.contains("basicNonLin")) {
    conf.basicnlin = BasicNLinConf(*tbl["basicNonLin"].as_table());
  }
  if (tbl.contains("kuramoto")) {
    conf.kuramoto = KuramotoConf(*tbl["kuramoto"].as_table());
  }
  spdlog::debug("Exiting tomlToDynConf.");
  return conf;
}

#undef SET_STRUCT_FIELD

// VectorXcd rhs(const VectorXcd& x, const SparseMatrix<c64>& J) { return J * x;
// }

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

static Eigen::IOFormat oneliner(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                " ", " ", "", "", "", "");
auto basicNonLin(const SparseMatrix<c64>& iH, f64 alpha) {
  return [&iH, alpha](const VectorXcd& x) {
    return VectorXcd(iH * x + alpha * x.cwiseAbs2());
  };
}

struct StateAndReservoir {
  typedef StateAndReservoir Self;
  VectorXcd x;
  VectorXd n;

  // StateAndReservoir operator*(const f64& b) const;
  Self& operator*=(f64 b) {
    x *= b;
    n *= b;
    return *this;
  }
  friend Self operator*(Self lhs, f64 b) { return lhs *= b; }
  Self& operator+=(const Self& rhs) {
    x += rhs.x;
    n += rhs.n;
    return *this;
  }
  friend Self operator+(Self lhs, const Self& b) { return lhs += b; }
  Self& operator-=(const Self& rhs) {
    x -= rhs.x;
    n -= rhs.n;
    return *this;
  }
  friend Self operator-(Self lhs, const Self& rhs) { return lhs -= rhs; }
};

auto coupledNonLin(const SparseMatrix<c64>& iH, const VectorXd& P, f64 alpha,
                   f64 R, f64 gamma, f64 Gamma) {
  return [&, alpha, R, gamma, Gamma](const StateAndReservoir& x) {
    return StateAndReservoir{iH * x.x + alpha * x.x.cwiseAbs2() + R * x.n -
                                 gamma * x.x,
                             P - Gamma * x.n};
  };
}

auto tetmNonLin(const SparseMatrix<f64>& iJ, const SparseMatrix<c64>& iL, f64 p,
                f64 alpha) {
  return [&, p, alpha](const MatrixX2cd& psi) {
    spdlog::debug("Function: tetmNonLin.");
    spdlog::debug("Making first matrix");
    MatrixX2cd first = (p - 1) * psi;
    u32 n = first.rows();
    u32 m = first.cols();
    spdlog::debug("First matrix has dims {}x{}", n, m);
    spdlog::debug("Making second matrix");
    MatrixX2cd second = -c64{0, alpha} * psi.cwiseAbs2().cwiseProduct(psi);
    n = second.rows();
    m = second.cols();
    spdlog::debug("Second matrix has dims {}x{}", n, m);
    spdlog::debug("Making third matrix");
    MatrixX2cd third = iJ * psi;
    n = third.rows();
    m = third.cols();
    spdlog::debug("Third matrix has dims {}x{}", n, m);
    spdlog::debug("Making fourth matrix");
    MatrixX2cd fourth(psi.rows(), 2);
    fourth.col(0) = iL * psi.col(1);
    fourth.col(1) = -iL.conjugate() * psi.col(0);
    n = fourth.rows();
    m = fourth.cols();
    spdlog::debug("Fourth matrix has dims {}x{}", n, m);
    spdlog::debug("Exiting tetmNonLin.");
    return first + second + third + fourth;
  };
}

MatrixX2cd explTETM(MatrixX2cd psi, const SparseMatrix<c64>& J,
                    const SparseMatrix<c64>& L, f64 p, f64 alpha) {
  MatrixX2cd complicated(psi.rows(), 2);
  complicated.col(0) = L * psi.col(1);
  complicated.col(1) = -L.conjugate() * psi.col(0);
  return p * psi - c64{1, alpha} * psi.cwiseAbs2().cwiseProduct(psi) + J * psi +
         complicated;
  // +complicated;
  // iJ * psi;
  // - c64{0, alpha} *  +
  //        iJ * psi + complicated;
}

MatrixX2cd tetmRK4Step(MatrixX2cd psi, const SparseMatrix<c64>& J,
                       const SparseMatrix<c64>& L, f64 p, f64 alpha, f64 dt) {

  MatrixX2cd k1 = explTETM(psi, J, L, p, alpha);
  MatrixX2cd k2 = explTETM(psi + 0.5 * dt * k1, J, L, p, alpha);
  MatrixX2cd k3 = explTETM(psi + 0.5 * dt * k2, J, L, p, alpha);
  MatrixX2cd k4 = explTETM(psi + dt * k3, J, L, p, alpha);
  MatrixX2cd ret = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
  return ret;
}

auto kuramoto(f64 K, u32 N, const VectorXd& omega) {
  return [&, K, N](const VectorXd& theta) {
    MatrixXd sins = (VectorXd::Ones(N) * theta.transpose() -
                     theta * VectorXd::Ones(N).transpose())
                        .array()
                        .sin();
    return VectorXd(omega + (K / N) * (sins).rowwise().sum());
  };
}

auto advancedKuramoto(f64 K, const VectorXd& omega) {
  return [&, K](const VectorXd& theta) {
    MatrixXd sins = (VectorXd::Ones(theta.size()) * theta.transpose() -
                     theta * VectorXd::Ones(theta.size()).transpose())
                        .array()
                        .sin();
    return VectorXd(omega +
                    (1.0 / (theta.size())) * (K * sins).rowwise().sum());
  };
}

auto basic(const SparseMatrix<c64>& iH) {
  return [&](const VectorXcd& x) { return VectorXcd(iH * x); };
}

auto Lorenz(f64 sigma, f64 R, f64 b) {
  return [=](Eigen::Vector3d x) {
    return Eigen::Vector3d(sigma * (x[1] - x[0]), R * x[0] - x[1] - x[0] * x[2],
                           -b * x[2] + x[0] * x[1]);
  };
}

auto LorenzDiff(f64 sigma, f64 R, f64 b) {
  return [=](Eigen::Vector3d x, Eigen::Vector3d p) {
    return Eigen::Vector3d(sigma * (p[1] - p[0]),
                           (R - x[2]) * p[0] - p[1] - x[0] * p[2],
                           -b * p[2] + p[0] * x[1] + x[0] * p[1]);
  };
}

auto basicHankel(f64 p, f64 alpha, f64 j, f64 sep) {
  return [=](Eigen::Vector2cd psi) {
    return Eigen::Vector2cd(
        p * psi - c64{1, alpha} * psi.cwiseAbs2().cwiseProduct(psi) +
        j * c64{gsl_sf_bessel_J0(sep), gsl_sf_bessel_Y0(sep)} *
            psi(Eigen::indexing::lastN(2).reverse()));
  };
}

auto noCoupling(f64 p, f64 alpha) {
  return
      [=](Eigen::Vector2cd psi) { return Eigen::Vector2cd(c64{0, -1} * psi); };
}

auto noCouplingDrivenDissipative(f64 p, f64 alpha) {
  return [=](Eigen::Vector2cd psi) {
    return Eigen::Vector2cd(p * psi -
                            c64{0, alpha} * psi.cwiseAbs2().cwiseProduct(psi));
  };
}

int doLorenz(const BasicConf&) {
  const f64 sigma = 10.0;
  const f64 R = 28.0;
  const f64 b = 8.0 / 3.0;

  const auto rhs = Lorenz(sigma, R, b);

  Eigen::Vector3d init{10.0, 10.0, 5.0};

  for (u64 i = 0; i < 1000; ++i) {
    init = rk4step(init, 0.01, rhs);
  }

  std::vector<Eigen::Vector3d> states(4, init);
  std::vector<Eigen::Vector3d> perturbations{
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  for (u64 i = 0; i < 100; ++i) {
  }
  return 0;
}

int doNoCoupling(const BasicDistanceConf& conf) {
  spdlog::debug("Function: doDistanceScan");
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  const auto seps = linspace(conf.sep, true);
  const Eigen::VectorXd times = linspace(conf.t, true);
  std::vector<c64> psil(conf.t.n);
  std::vector<c64> psir(conf.t.n);
  spdlog::debug("Entering separation loop.");
  const auto rhs = noCoupling(conf.p, conf.alpha);
  const f64 seed1 = dis(gen);
  const f64 seed2 = dis(gen);
  Eigen::Vector2cd psi{c64{cos(seed1), sin(seed1)},
                       c64{cos(seed2), sin(seed2)}};
  psil[0] = psi.x();
  psir[0] = psi.y();
  spdlog::debug("Entering solution loop.");
  for (u64 j = 1; j < conf.t.n; ++j) {
    psi = rk4step(psi, conf.t.d(), rhs);
    psil[j] = psi.x();
    psir[j] = psi.y();
  }
  spdlog::debug("Finished calculating.");

  H5File file(conf.outfile.c_str());
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }
  spdlog::debug("Writing psil to file.");
  writeArray<1>("psil", file, c_double_id, psil.data(), {conf.t.n});
  spdlog::debug("Writing psir to file.");
  writeArray<1>("psir", file, c_double_id, psir.data(), {conf.t.n});
  spdlog::debug("Writing time to file.");
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g, (void*)times.data(),
                {conf.t.n});
  spdlog::debug("Exiting doNoCoupling.");
  return 0;
}

int doNCDD(const BasicDistanceConf& conf) {
  spdlog::debug("Function: doDistanceScan");
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  const auto seps = linspace(conf.sep, true);
  const Eigen::VectorXd times = linspace(conf.t, true);
  std::vector<c64> psil(conf.t.n);
  std::vector<c64> psir(conf.t.n);
  spdlog::debug("Entering separation loop.");
  const auto rhs = noCouplingDrivenDissipative(conf.p, conf.alpha);
  const f64 seed1 = dis(gen);
  const f64 seed2 = dis(gen);
  Eigen::Vector2cd psi{c64{cos(seed1), sin(seed1)},
                       c64{cos(seed2), sin(seed2)}};
  psil[0] = psi.x();
  psir[0] = psi.y();
  spdlog::debug("Entering solution loop.");
  for (u64 j = 1; j < conf.t.n; ++j) {
    psi = rk4step(psi, conf.t.d(), rhs);
    psil[j] = psi.x();
    psir[j] = psi.y();
  }
  spdlog::debug("Finished calculating.");

  H5File file(conf.outfile.c_str());
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }
  spdlog::debug("Writing psil to file.");
  writeArray<1>("psil", file, c_double_id, psil.data(), {conf.t.n});
  spdlog::debug("Writing psir to file.");
  writeArray<1>("psir", file, c_double_id, psir.data(), {conf.t.n});
  spdlog::debug("Writing time to file.");
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g, (void*)times.data(),
                {conf.t.n});
  spdlog::debug("Exiting doDistanceScan.");
  return 0;
}

int doDistanceScan(const BasicDistanceConf& conf) {
  spdlog::debug("Function: doDistanceScan");
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  const auto seps = linspace(conf.sep, true);
  const Eigen::VectorXd times = linspace(conf.t, true);
  std::vector<c64> psil(conf.t.n * conf.sep.n);
  std::vector<c64> psir(conf.t.n * conf.sep.n);
  spdlog::debug("Entering separation loop.");
#pragma omp parallel for
  for (u64 i = 0; i < conf.sep.n; ++i) {
    const auto rhs = basicHankel(conf.p, conf.alpha, conf.j, conf.sep.ith(i));
    const f64 seed1 = dis(gen);
    const f64 seed2 = dis(gen);
    Eigen::Vector2cd psi{c64{cos(seed1), sin(seed1)},
                         c64{cos(seed2), sin(seed2)}};
    psil[i * conf.t.n] = psi.x();
    psir[i * conf.t.n] = psi.y();
    spdlog::debug("Entering solution loop.");
    for (u64 j = 1; j < conf.t.n; ++j) {
      psi = rk4step(psi, conf.t.d(), rhs);
      psil[i * conf.t.n + j] = psi.x();
      psir[i * conf.t.n + j] = psi.y();
    }
  }
  spdlog::debug("Finished calculating.");

  H5File file(conf.outfile.c_str());
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }
  spdlog::debug("Writing psil to file.");
  writeArray<2>("psil", file, c_double_id, psil.data(), {conf.sep.n, conf.t.n});
  spdlog::debug("Writing psir to file.");
  writeArray<2>("psir", file, c_double_id, psir.data(), {conf.sep.n, conf.t.n});
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g, (void*)times.data(),
                {conf.t.n});
  spdlog::debug("Exiting doDistanceScan.");
  return 0;
}

auto hankelDD(const SparseMatrix<c64>& J, f64 p, f64 alpha) {
  return [&, p, alpha](Eigen::VectorXcd psi) {
    return p * psi - c64{1, alpha} * psi.cwiseAbs2().cwiseProduct(psi) +
           J * psi;
  };
}

typedef struct {
  f64 p;
  f64 alpha;
  f64 j;
  f64 rscale;
} Params;

HighFive::CompoundType compoundParams() {
  return {{"p", HighFive::create_datatype<f64>()},
          {"alpha", HighFive::create_datatype<f64>()},
          {"j", HighFive::create_datatype<f64>()},
          {"rscale", HighFive::create_datatype<f64>()}};
}

HIGHFIVE_REGISTER_TYPE(Params, compoundParams);

int doBasicHankelDD(const TETMConf& conf) {
  spdlog::debug("Function: doBasicHankelDD.");
  HighFive::File pc(conf.pointPath, HighFive::File::ReadOnly);
  spdlog::debug("Read pointfile.");
  Eigen::MatrixX2d points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
  spdlog::debug("Read points.");
  Eigen::MatrixX2i couplings =
      pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
  spdlog::debug("Read couplings.");
  const f64 rscale = conf.rscale;
  const SparseMatrix<c64> J =
      conf.j * SparseC(points, couplings, [rscale](Vector2d d) {
        return c64{gsl_sf_bessel_J0(rscale * d.norm()),
                   gsl_sf_bessel_Y0(rscale * d.norm())};
      });
  spdlog::debug("Made sparse matrix J.");
  auto sol = Eigen::ComplexEigenSolver<MatrixXcd>(MatrixXcd(J));
  auto max_coeff = std::ranges::max_element(
      sol.eigenvalues(), [](c64 a, c64 b) { return a.imag() < b.imag(); });
  std::cout << "Highest eigenvalue is: " << max_coeff->imag() << '\n';
  const f64 p = (conf.p - 1) * max_coeff->imag();
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  spdlog::debug("Allocating psi.");
  VectorXcd psi(points.rows());
  u64 n = psi.rows();
  u64 m = psi.cols();
  spdlog::debug("Allocated psi with dims {}x{}", n, m);
  spdlog::debug("Writing random coordinates to psi.");
  for (u64 i = 0; i < psi.size(); i++) {
    auto x = dis(gen);
    *(psi.data() + i) = {1e-4 * cos(x), 1e-4 * sin(x)};
  }
  const u32 overallSize = conf.t.n * psi.size();
  spdlog::debug("Allocating psipdata with {} elements.", overallSize);
  MatrixXcd psidata(psi.size(), conf.t.n + 1);
  const size_t byteSize = psi.size() * sizeof(c64);
  psidata(Eigen::indexing::all, 0) = psi;
  std::cout << "byteSize is: " << byteSize << '\n';
  std::cout << "Byte size of psidata is: " << psidata.size() * sizeof(c64);
  auto rhs = hankelDD(J, p, conf.alpha);
  std::cout << conf.t.n << '\n';
  for (u32 i = 1; i < conf.t.n + 1; i++) {
    psi = rk4step(psi, conf.t.d(), rhs);
    n = psi.rows();
    m = psi.cols();
    spdlog::debug("Return psiish with dims {}x{}", n, m);
    spdlog::debug("Copying to psipdata.");
    psidata(Eigen::indexing::all, i) = psi;
  }
  spdlog::debug("Finished calculating.");

  HighFive::File file(conf.outfile, HighFive::File::Truncate);
  spdlog::debug("Writing psip to file.");

  file.createDataSet("psi", psidata);
  file.createDataSet("points", points);
  file.createDataSet("couplings", couplings);
  file.createDataSet("time", linspace(conf.t, true));
  HighFive::DataSet paramSet = file.createDataSet(
      "params", Params{conf.p, conf.alpha, conf.j, conf.rscale});
  spdlog::debug("Exiting doTETM.");
  return 0;
}

int doHankelScan(const std::vector<HankelScanConf>& confs) {
  spdlog::debug("Function: doHankelScan.");
  for (const auto& conf : confs) {
    HighFive::File pc(conf.pointPath, HighFive::File::ReadOnly);
    spdlog::debug("Read pointfile.");
    Eigen::MatrixX2d points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    spdlog::debug("Read points.");
    Eigen::MatrixX2i couplings =
        pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
    spdlog::debug("Read couplings.");

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(0.0, 2 * M_PI);

    s64 samples = conf.ps.n * conf.alphas.n * conf.js.n * conf.rscales.n;
    std::vector<c64> data(samples * points.rows());
    s64 datasize = data.size();
    spdlog::debug("data has {} elements in total.", datasize);
    s64 psize = conf.ps.n;
    spdlog::debug("number of ps is {}.", psize);
    s64 alphasize = conf.alphas.n;
    spdlog::debug("number of alphas is {}.", alphasize);
    s64 jsize = conf.js.n;
    spdlog::debug("number of js is {}.", jsize);
    s64 rscalesize = conf.rscales.n;
    spdlog::debug("number of rscales is {}.", rscalesize);
    VectorXcd init_psi(points.rows());
    for (u64 o = 0; o < init_psi.size(); ++o) {
      auto x = dis(gen);
      *(init_psi.data() + o) = {1e-4 * cos(x), 1e-4 * sin(x)};
    }
    for (s64 m = 0; m < jsize; ++m) {
      const f64 j = conf.js.ith(m);
#pragma omp parallel for
      for (s64 n = 0; n < rscalesize; ++n) {
        const f64 scale = conf.rscales.ith(n);
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
          const f64 p = conf.ps.ith(k);
          for (s64 l = 0; l < alphasize; ++l) {
            const f64 alpha = conf.alphas.ith(l);
            const f64 eff_p = (p - 1) * min_coeff->imag();
            spdlog::debug("Made sparse matrix J.");
            spdlog::debug("Allocating psi.");
            VectorXcd psi = init_psi;
            spdlog::debug("Writing random coordinates to psi.");
            const size_t byteSize = psi.size() * sizeof(c64);
            auto rhs = hankelDD(J, eff_p, alpha);
            spdlog::debug("Running rk4 for {} steps.", conf.t.n);
            for (u32 o = 1; o < conf.t.n + 1; ++o) {
              psi = rk4step(psi, conf.t.d(), rhs);
            }
            s64 idx = (n + rscalesize * (m + jsize * (l + alphasize * k))) *
                      points.rows();
            spdlog::debug("Copying to {}-th complex number.", idx);
            spdlog::debug("Finished calculating.");
            memcpy(data.data() + idx, psi.data(), byteSize);
          }
        }
      }
    }

    HighFive::File file(conf.outfile, HighFive::File::Truncate);
    spdlog::debug("Writing samples to file.");

    auto sampleSet = file.createDataSet<c64>(
        "psis", HighFive::DataSpace(
                    {static_cast<u64>(psize), static_cast<u64>(alphasize),
                     static_cast<u64>(jsize), static_cast<u64>(rscalesize),
                     static_cast<u64>(points.rows())}));
    sampleSet.write_raw(data.data());
    file.createDataSet("points", points);
    file.createDataSet("couplings", couplings);
    file.createDataSet("time", linspace(conf.t, true));
    file.createDataSet("ps", linspace(conf.ps, false));
    file.createDataSet("alphas", linspace(conf.alphas, false));
    file.createDataSet("js", linspace(conf.js, false));
    file.createDataSet("rscales", linspace(conf.rscales, false));
  }
  spdlog::debug("Exiting doHankelScan.");
  return 0;
}

int doHankelTimeScan(const std::vector<HankelScanConf>& confs) {
  spdlog::debug("Function: doHankelScan.");
  for (const auto& conf : confs) {
    HighFive::File pc(conf.pointPath, HighFive::File::ReadOnly);
    spdlog::debug("Read pointfile.");
    Eigen::MatrixX2d points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    spdlog::debug("Read points.");
    Eigen::MatrixX2i couplings =
        pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
    spdlog::debug("Read couplings.");

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(0.0, 2 * M_PI);

    s64 samples = conf.ps.n * conf.alphas.n * conf.js.n * conf.rscales.n;
    std::vector<c64> data(samples * 2 * (conf.t.n + 1));
    s64 datasize = data.size();
    std::vector<c64> snapshotdata(samples * points.rows());
    s64 snapshotsize = snapshotdata.size();
    spdlog::debug("data has {} elements in total.", datasize);
    s64 psize = conf.ps.n;
    spdlog::debug("number of ps is {}.", psize);
    s64 alphasize = conf.alphas.n;
    spdlog::debug("number of alphas is {}.", alphasize);
    s64 jsize = conf.js.n;
    spdlog::debug("number of js is {}.", jsize);
    s64 rscalesize = conf.rscales.n;
    spdlog::debug("number of rscales is {}.", rscalesize);
    VectorXcd init_psi(points.rows());
    for (u64 o = 0; o < init_psi.size(); ++o) {
      auto x = dis(gen);
      *(init_psi.data() + o) = {1e-4 * cos(x), 1e-4 * sin(x)};
    }
    const size_t byteSize = init_psi.size() * sizeof(c64);
    s64 idx = 0;
    for (s64 m = 0; m < jsize; ++m) {
      const f64 j = conf.js.ith(m);
      for (s64 n = 0; n < rscalesize; ++n) {
        const f64 scale = conf.rscales.ith(n);
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
          const f64 p = conf.ps.ith(k);
          for (s64 l = 0; l < alphasize; ++l) {
            const f64 alpha = conf.alphas.ith(l);
            const f64 eff_p = (p - 1) * min_coeff->imag();
            spdlog::debug("Made sparse matrix J.");
            spdlog::debug("Allocating psi.");
            VectorXcd psi = init_psi;
            spdlog::debug("Writing random coordinates to psi.");
            // const size_t byteSize = psi.size() * sizeof(c64);
            auto rhs = hankelDD(J, eff_p, alpha);
            spdlog::debug("Running rk4 for {} steps.", conf.t.n);
            c64 psisum = psi.sum();
            data[idx] = psisum;
            data[idx + 1] = psi[0];
            idx += 2;
            for (u32 o = 1; o < conf.t.n + 1; ++o) {
              psi = rk4step(psi, conf.t.d(), rhs);
              c64 psisum = psi.sum();
              data[idx] = psisum;
              data[idx + 1] = psi[0];
              idx += 2;
            }
            s64 idx = (n + rscalesize * (m + jsize * (l + alphasize * k))) *
                      points.rows();
            memcpy(snapshotdata.data() + idx, psi.data(), byteSize);
          }
        }
      }
    }

    HighFive::File file(conf.outfile, HighFive::File::Truncate);
    spdlog::debug("Writing samples to file.");

    auto seriesSet = file.createDataSet<c64>(
        "sumpsitimeseries",
        HighFive::DataSpace(
            {static_cast<u64>(psize), static_cast<u64>(alphasize),
             static_cast<u64>(jsize), static_cast<u64>(rscalesize),
             static_cast<u64>(conf.t.n + 1), 2}));
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
    file.createDataSet("time", linspace(conf.t, true));
    file.createDataSet("ps", linspace(conf.ps, false));
    file.createDataSet("alphas", linspace(conf.alphas, false));
    file.createDataSet("js", linspace(conf.js, false));
    file.createDataSet("rscales", linspace(conf.rscales, false));
  }
  spdlog::debug("Exiting doHankelTimeScan.");
  return 0;
}

int GPUHankelTimeScan(const std::vector<HankelScanConf>& confs) {
  spdlog::debug("Function: doHankelScan.");
  for (const auto& conf : confs) {
    HighFive::File pc(conf.pointPath, HighFive::File::ReadOnly);
    spdlog::debug("Read pointfile.");
    Eigen::MatrixX2d points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    spdlog::debug("Read points.");
    Eigen::MatrixX2i couplings =
        pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
    spdlog::debug("Read couplings.");

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(0.0, 2 * M_PI);

    s64 samples = conf.ps.n * conf.alphas.n * conf.js.n * conf.rscales.n;
    std::vector<c64> data(samples * 2 * (conf.t.n + 1));
    s64 datasize = data.size();
    std::vector<c64> snapshotdata(samples * points.rows());
    s64 snapshotsize = snapshotdata.size();
    spdlog::debug("data has {} elements in total.", datasize);
    s64 psize = conf.ps.n;
    spdlog::debug("number of ps is {}.", psize);
    s64 alphasize = conf.alphas.n;
    spdlog::debug("number of alphas is {}.", alphasize);
    s64 jsize = conf.js.n;
    spdlog::debug("number of js is {}.", jsize);
    s64 rscalesize = conf.rscales.n;
    spdlog::debug("number of rscales is {}.", rscalesize);
    VectorXcd init_psi(points.rows());
    for (u64 o = 0; o < init_psi.size(); ++o) {
      auto x = dis(gen);
      *(init_psi.data() + o) = {1e-4 * cos(x), 1e-4 * sin(x)};
    }
    const size_t byteSize = init_psi.size() * sizeof(c64);
    s64 idx = 0;
    for (s64 m = 0; m < jsize; ++m) {
      const f64 j = conf.js.ith(m);
      for (s64 n = 0; n < rscalesize; ++n) {
        const f64 scale = conf.rscales.ith(n);
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
          const f64 p = conf.ps.ith(k);
          for (s64 l = 0; l < alphasize; ++l) {
            const f64 alpha = conf.alphas.ith(l);
            const f64 eff_p = (p - 1) * min_coeff->imag();
            spdlog::debug("Made sparse matrix J.");
            spdlog::debug("Allocating psi.");
            VectorXcd psi = init_psi;
            spdlog::debug("Writing random coordinates to psi.");
            // const size_t byteSize = psi.size() * sizeof(c64);
            auto rhs = hankelDD(J, eff_p, alpha);
            spdlog::debug("Running rk4 for {} steps.", conf.t.n);
            c64 psisum = psi.sum();
            data[idx] = psisum;
            data[idx + 1] = psi[0];
            idx += 2;
            for (u32 o = 1; o < conf.t.n + 1; ++o) {
              psi = rk4step(psi, conf.t.d(), rhs);
              c64 psisum = psi.sum();
              data[idx] = psisum;
              data[idx + 1] = psi[0];
              idx += 2;
            }
            s64 idx = (n + rscalesize * (m + jsize * (l + alphasize * k))) *
                      points.rows();
            memcpy(snapshotdata.data() + idx, psi.data(), byteSize);
          }
        }
      }
    }

    HighFive::File file(conf.outfile, HighFive::File::Truncate);
    spdlog::debug("Writing samples to file.");

    auto seriesSet = file.createDataSet<c64>(
        "sumpsitimeseries",
        HighFive::DataSpace(
            {static_cast<u64>(psize), static_cast<u64>(alphasize),
             static_cast<u64>(jsize), static_cast<u64>(rscalesize),
             static_cast<u64>(conf.t.n + 1), 2}));
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
    file.createDataSet("time", linspace(conf.t, true));
    file.createDataSet("ps", linspace(conf.ps, false));
    file.createDataSet("alphas", linspace(conf.alphas, false));
    file.createDataSet("js", linspace(conf.js, false));
    file.createDataSet("rscales", linspace(conf.rscales, false));
  }
  spdlog::debug("Exiting doHankelTimeScan.");
  return 0;
}

int doBasic(const BasicConf& conf) {
  std::vector<Point> points = readPoints(conf.pointPath);
  kdt::KDTree<Point> kdtree(points);
  f64 radius = conf.searchRadius.has_value() ? conf.searchRadius.value()
                                             : 1.01 * avgNNDist(kdtree, points);
  if (!conf.searchRadius.has_value()) {
    std::cout << "Automatically determined search radius: " << radius << '\n';
  }
  SparseMatrix<c64> iH =
      c64{0, -1} * SparseH(points, kdtree, radius,
                           [](Vector2d d) { return std::exp(-d.norm()); });
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  VectorXcd psi(points.size());
  for (auto& e : psi) {
    e = {cosf(dis(gen)), sinf(dis(gen))};
  }
  auto rhs = basic(iH);

  std::vector<c64> orderparam(conf.t.n + 1);
  orderparam[0] =
      (c64{0, 1} * psi.array().arg()).exp().sum() / (f64)points.size();
  for (u32 i = 1; i < conf.t.n + 1; i++) {
    psi = rk4step(psi, conf.t.d(), rhs);
    orderparam[i] =
        (c64{0, 1} * psi.array().arg()).exp().sum() / (f64)points.size();
    // memcpy(outdata.data() + i * points.size(), psi.data(), points.size());
  }
  H5File file(conf.outfile.c_str());
  // hid_t file =
  //     H5Fcreate(conf.outfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
  //     H5P_DEFAULT);
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }
  std::cout << "Writing full data\n";
  writeArray<1>("data", file, c_double_id, orderparam.data(), {conf.t.n + 1});
  std::cout << "Writing corresponding times\n";
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g,
                linspace(conf.t, true).data(), {conf.t.n});

  return 0;
}

int doTETM(const TETMConf& conf) {
  spdlog::debug("Function: doTETM.");
  const std::vector<Point> points = readPoints(conf.pointPath);
  spdlog::debug("Read points.");
  kdt::KDTree<Point> kdtree(points);
  spdlog::debug("Built kdtree");
  const f64 avgradius = avgNNDist(kdtree, points);
  const f64 radius = conf.searchRadius.has_value() ? conf.searchRadius.value()
                                                   : 1.01 * avgradius;
  std::cout << "Radius is: " << radius << '\n';
  spdlog::debug("Using search radius: {}", radius);
  SparseMatrix<c64> J =
      conf.j * SparseHC(points, kdtree, radius, [](Vector2d d) {
        return c64{gsl_sf_bessel_J0(d.norm()), gsl_sf_bessel_Y0(d.norm())};
      });
  spdlog::debug("Made sparse matrix iJ.");
  auto sol = hermitianEigenSolver(MatrixXcd(J));
  std::cout << "Highest eigenvalue is: " << sol.D.maxCoeff() << '\n';
  SparseMatrix<c64> L =
      0.5 * conf.j * SparseHC(points, kdtree, radius, [](Vector2d d) {
        // std::cout << d.squaredNorm() << '\n';
        return c64{gsl_sf_bessel_J0(d.norm()), gsl_sf_bessel_Y0(d.norm())} *
               c64{1 - 2 * d(1) * d(1) / d.squaredNorm(),
                   2 * d(0) * d(1) / d.squaredNorm()};
      });
  // std::cout << L << '\n';
  spdlog::debug("Made sparse matrix iL.");
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  spdlog::debug("Allocating psi.");
  MatrixX2cd psi(points.size(), 2);
  u64 n = psi.rows();
  u64 m = psi.cols();
  spdlog::debug("Allocated psi with dims {}x{}", n, m);
  spdlog::debug("Writing random coordinates to psi.");
  for (u64 i = 0; i < psi.size(); i++) {
    auto x = dis(gen);
    *(psi.data() + i) = {cos(x), sin(x)};
  }
  u32 overallSize = conf.t.n * points.size();
  spdlog::debug("Allocating psipdata with {} elements.", overallSize);
  std::vector<c64> psidata((conf.t.n) * points.size() * 2);
  size_t byteSize = psi.size() * sizeof(c64);
  memcpy(psidata.data(), psi.data(), byteSize);
  // //dlog::debug(std::cout, "Allocating psimdata with {} elements.",
  // overallSize); std::vector<c64> psimdata(conf.t.n * points.size());
  // //dlog::debug(std::cout, "Testing allocation first");
  // MatrixX2cd first = (conf.p - 1) * psi;
  // //dlog::debug(std::cout, "Testing allocation second");
  // MatrixX2cd second = -c64{0, conf.alpha} * psi.cwiseAbs2().transpose() *
  // psi; //dlog::debug(std::cout, "Testing allocation third"); MatrixX2cd third
  // = iJ * psi; //dlog::debug(std::cout, "Testing allocation fourth");
  // MatrixX2cd fourth(psi.rows(), 2); //dlog::debug(std::cout, "Testing spm v
  // multiplication"); MatrixXcd b = iL * psi.col(1); //dlog::debug(std::cout,
  // "Setting other fourth values"); fourth.col(1) = -iL.conjugate() *
  // psi.col(0); auto rhs = tetmNonLin(iJ, iL, conf.p, conf.alpha);
  // //dlog::debug(std::cout, "Created rhs function.");
  std::cout << "byteSize is: " << byteSize << '\n';
  std::cout << "Byte size of psidata is: " << psidata.size() * sizeof(c64);
  for (u32 i = 1; i < conf.t.n; i++) {
    psi = tetmRK4Step(psi, J, L, conf.p, conf.alpha, conf.t.d());
    u64 n = psi.rows();
    u64 m = psi.cols();
    spdlog::debug("Return psiish with dims {}x{}", n, m);
    spdlog::debug("Copying to psipdata.");
    memcpy(psidata.data() + i * psi.size(), psi.data(), byteSize);
    // //dlog::debug(std::cout, "Copying to psimdata.");
    // memcpy(psimdata.data() + i * byteSize, psi.data(), byteSize);
  }
  spdlog::debug("Finished calculating.");

  H5File file(conf.outfile.c_str());
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }
  spdlog::debug("Writing psip to file.");
  writeArray<3>("psi", file, c_double_id, psidata.data(),
                {conf.t.n, 2, points.size()});
  // //dlog::debug(std::cout, "Writing psim to file.");
  // writeArray<2>("psip", file, c_double_id, psipdata.data(),
  //               {conf.t.n, points.size()});
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g,
                linspace(conf.t, true).data(), {conf.t.n});
  spdlog::debug("Exiting doTETM.");
  return 0;
}

struct SpecConsts {
  u32 len;
  f32 dt;
  f32 alpha;
};

int GPUDrivenDiss(const std::vector<HankelScanConf>& confs) {
  spdlog::debug("Function: GPUDrivenDiss.");
  spdlog::debug("Constructing manager.");
  Manager mgr(100 * 1024 * 1024);
  for (const auto& conf : confs) {
    HighFive::File pc(conf.pointPath, HighFive::File::ReadOnly);
    spdlog::debug("Read pointfile.");
    Eigen::MatrixX2d points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    spdlog::debug("Read points.");
    Eigen::MatrixX2i couplings =
        pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
    spdlog::debug("Read couplings.");

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<f32> dis(0.0, 2 * M_PI);

    spdlog::debug("allocating gpu_psi");
    MetaBuffer gpu_psi = mgr.makeRawBuffer<c32>(points.rows());
    spdlog::debug("allocating gpu_J");
    MetaBuffer gpu_J = mgr.makeRawBuffer<c32>(couplings.rows() * 2);
    spdlog::debug("allocating gpu_ps");
    MetaBuffer gpu_ps = mgr.makeRawBuffer<f32>(points.rows());
    spdlog::debug("allocating gpu_k1");
    MetaBuffer gpu_k1 = mgr.makeRawBuffer<c32>(points.rows());
    spdlog::debug("allocating gpu_k2");
    MetaBuffer gpu_k2 = mgr.makeRawBuffer<c32>(points.rows());
    spdlog::debug("allocating gpu_k3");
    MetaBuffer gpu_k3 = mgr.makeRawBuffer<c32>(points.rows());
    spdlog::debug("allocating gpu_k4");
    MetaBuffer gpu_k4 = mgr.makeRawBuffer<c32>(points.rows());
    std::vector<u32> col_indices(couplings.rows() * 2);
    std::vector<u32> row_indices(points.rows() + 1);
    {
      const SparseMatrix<c64> bleh =
          SparseC(points, couplings, [](Vector2d d) { return c64{1, 0}; });
      for (u32 i = 0; i < points.rows() + 1; ++i) {
        spdlog::debug("Sparse Matrix row index {}: {}", i,
                      bleh.outerIndexPtr()[i]);
        row_indices[i] = static_cast<u32>(bleh.outerIndexPtr()[i]);
      }
      for (u32 i = 0; i < 2 * couplings.rows(); ++i) {
        spdlog::debug("Sparse Matrix col index {}: {}", i,
                      bleh.innerIndexPtr()[i]);
        col_indices[i] = static_cast<u32>(bleh.innerIndexPtr()[i]);
      }
    }
    spdlog::debug("moving col_indices to gpu");
    MetaBuffer gpu_col_indices = mgr.vecToBuffer(col_indices);
    spdlog::debug("moving row_indices to gpu");
    MetaBuffer gpu_row_indices = mgr.vecToBuffer(row_indices);
    s64 samples = conf.ps.n * conf.alphas.n * conf.js.n * conf.rscales.n;
    spdlog::debug("Taking {} samples", samples);
    std::vector<c32> data(samples * points.rows());
    s64 datasize = data.size();
    spdlog::debug("data has {} elements in total.", datasize);
    s64 psize = conf.ps.n;
    spdlog::debug("number of ps is {}.", psize);
    s64 alphasize = conf.alphas.n;
    spdlog::debug("number of alphas is {}.", alphasize);
    s64 jsize = conf.js.n;
    spdlog::debug("number of js is {}.", jsize);
    s64 rscalesize = conf.rscales.n;
    spdlog::debug("number of rscales is {}.", rscalesize);
    for (s64 m = 0; m < jsize; ++m) {
      const f64 j = conf.js.ith(m);
      for (s64 n = 0; n < rscalesize; ++n) {
        const f64 scale = conf.rscales.ith(n);
        const SparseMatrix<c32> J =
            j * SparseCf(points, couplings, [scale](Vector2d d) {
              return c32{static_cast<f32>(gsl_sf_bessel_J0(scale * d.norm())),
                         static_cast<f32>(gsl_sf_bessel_Y0(scale * d.norm()))};
            });
        mgr.writeToBuffer(gpu_J, J.valuePtr(), 2 * couplings.size());
        const auto sol = Eigen::ComplexEigenSolver<MatrixXcf>(MatrixXcf(J));
        auto max_coeff =
            std::ranges::max_element(sol.eigenvalues(), [](c64 a, c64 b) {
              return a.real() < b.real();
            });
        for (s64 l = 0; l < alphasize; ++l) {
          const f64 alpha = conf.alphas.ith(l);
          const SpecConsts spc{static_cast<u32>(points.rows()),
                               static_cast<f32>(conf.t.d()),
                               static_cast<f32>(alpha)};
          Algorithm alg = mgr.makeAlgorithm<u32>(
              "drivendiss.spv",
              {&gpu_psi, &gpu_k1, &gpu_k2, &gpu_k3, &gpu_k4, &gpu_ps, &gpu_J,
               &gpu_row_indices, &gpu_col_indices},
              spc);
          for (s64 k = 0; k < psize; ++k) {
            const f64 p = conf.ps.ith(k);
            const f32 eff_p = (p - 1) * max_coeff->real();
            std::vector<f32> ps(points.rows(), eff_p);
            mgr.writeToBuffer(gpu_ps, ps);
            spdlog::debug("Made sparse matrix J.");
            spdlog::debug("Allocating psi.");
            VectorXcf psi(points.rows());
            spdlog::debug("Writing random coordinates to psi.");
            for (u64 o = 0; o < psi.size(); ++o) {
              auto x = dis(gen);
              *(psi.data() + o) = {1e-4f * cos(x), 1e-4f * sin(x)};
            }
            mgr.writeToBuffer(gpu_psi, psi.data(), sizeof(c32) * psi.size());
            u32 super_steps = conf.t.n / 128;
            u32 end_steps = conf.t.n % 128;
            vk::CommandBuffer super_cb = mgr.beginRecord();
            vk::CommandBuffer end_cb = mgr.beginRecord();
            for (u32 q = 0; q < 16; ++q) {
              u32 stage = 0;
              super_cb.pushConstants(alg.m_PipelineLayout,
                                     vk::ShaderStageFlagBits::eCompute, 0, 4,
                                     &stage);
              appendOp(super_cb, alg, (psi.size() + 63) / 64, 1, 1);
              stage = 1;
              super_cb.pushConstants(alg.m_PipelineLayout,
                                     vk::ShaderStageFlagBits::eCompute, 0, 4,
                                     &stage);
              appendOp(super_cb, alg, (psi.size() + 63) / 64, 1, 1);
              stage = 2;
              super_cb.pushConstants(alg.m_PipelineLayout,
                                     vk::ShaderStageFlagBits::eCompute, 0, 4,
                                     &stage);
              appendOp(super_cb, alg, (psi.size() + 63) / 64, 1, 1);
              stage = 3;
              super_cb.pushConstants(alg.m_PipelineLayout,
                                     vk::ShaderStageFlagBits::eCompute, 0, 4,
                                     &stage);
              appendOp(super_cb, alg, (psi.size() + 63) / 64, 1, 1);
              stage = 4;
              super_cb.pushConstants(alg.m_PipelineLayout,
                                     vk::ShaderStageFlagBits::eCompute, 0, 4,
                                     &stage);
              appendOp(super_cb, alg, (psi.size() + 63) / 64, 1, 1);
            }
            {
              u32 stage = 0;
              end_cb.pushConstants(alg.m_PipelineLayout,
                                   vk::ShaderStageFlagBits::eCompute, 0, 4,
                                   &stage);
              appendOp(end_cb, alg, (psi.size() + 63) / 64, 1, 1);
              stage = 1;
              end_cb.pushConstants(alg.m_PipelineLayout,
                                   vk::ShaderStageFlagBits::eCompute, 0, 4,
                                   &stage);
              appendOp(end_cb, alg, (psi.size() + 63) / 64, 1, 1);
              stage = 2;
              end_cb.pushConstants(alg.m_PipelineLayout,
                                   vk::ShaderStageFlagBits::eCompute, 0, 4,
                                   &stage);
              appendOp(end_cb, alg, (psi.size() + 63) / 64, 1, 1);
              stage = 3;
              end_cb.pushConstants(alg.m_PipelineLayout,
                                   vk::ShaderStageFlagBits::eCompute, 0, 4,
                                   &stage);
              appendOp(end_cb, alg, (psi.size() + 63) / 64, 1, 1);
              stage = 4;
              end_cb.pushConstants(alg.m_PipelineLayout,
                                   vk::ShaderStageFlagBits::eCompute, 0, 4,
                                   &stage);
              appendOp(end_cb, alg, (psi.size() + 63) / 64, 1, 1);
            }
            super_cb.end();
            end_cb.end();
            for (u32 o = 0; o < super_steps; ++o) {
              mgr.execute(super_cb);
            }
            for (u32 o = 0; o < end_steps; ++o) {
              mgr.execute(end_cb);
            }
            const size_t byteSize = psi.size() * sizeof(c32);

            mgr.writeFromBuffer(gpu_psi, psi.data(), sizeof(c32) * psi.size());
            size_t idx = (n + rscalesize * (m + jsize * (l + alphasize * k))) *
                         points.rows();
            spdlog::debug("Copying to {}-th complex number.", idx);
            spdlog::debug("Finished calculating.");
            memcpy(data.data() + idx, psi.data(), byteSize);
          }
        }
      }
    }

    HighFive::File file(conf.outfile, HighFive::File::Truncate);
    spdlog::debug("Writing samples to file.");

    auto sampleSet = file.createDataSet<c32>(
        "psis", HighFive::DataSpace(
                    {static_cast<u64>(psize), static_cast<u64>(alphasize),
                     static_cast<u64>(jsize), static_cast<u64>(rscalesize),
                     static_cast<u64>(points.rows())}));
    sampleSet.write_raw(data.data());
    file.createDataSet("points", points);
    file.createDataSet("couplings", couplings);
    file.createDataSet("time", linspace(conf.t, true));
    file.createDataSet("ps", linspace(conf.ps, false));
    file.createDataSet("alphas", linspace(conf.alphas, false));
    file.createDataSet("js", linspace(conf.js, false));
    file.createDataSet("rscales", linspace(conf.rscales, false));
  }
  spdlog::debug("Exiting doHankelScan.");

  return 0;
}

// int GPUTETM(const TETMConf& conf) {
//   std::vector<Point> points = readPoints(conf.pointPath);
//   kdt::KDTree<Point> kdtree(points);
//   f64 radius = conf.searchRadius.has_value() ? conf.searchRadius.value()
//                                              : 1.01 * avgNNDist(kdtree,
//                                              points);
//   SparseMatrix<c64> iJ = SparseHC(points, kdtree, radius, [](Vector2d d) {
//     return c64{0, -1} * std::exp(-d.norm());
//   });
//   auto iL =
//       c64{0, -1} * SparseHC(points, kdtree, radius, [](Vector2d d) {
//         return std::exp(-d.norm()) * c64{1 - 2 * d(1) * d(1) /
//         d.squaredNorm(),
//                                          2 * d(0) * d(1) /
//                                          d.squaredNorm()};
//       });
//   std::random_device dev;
//   std::mt19937 gen(dev());
//   std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
//   MatrixX2cd psi(points.size(), 2);
//   for (u64 i = 0; i < psi.size(); i++) {
//     *(psi.data() + i) = {cos(dis(gen)), sin(dis(gen))};
//   }
//   std::vector<c64> psipdata(conf.t.n * points.size());
//   std::vector<c64> psimdata(conf.t.n * points.size());
//   auto rhs = tetmNonLin(iJ, iL, conf.p, conf.alpha);
//   for (u32 i = 0; i < conf.t.n; i++) {
//     psi = rk4step(psi, conf.t.d(), rhs);
//     size_t byteSize = psi.size() * sizeof(c64);
//     memcpy(psipdata.data() + i * byteSize, psi.data(), byteSize);
//   }
//
//   H5File file(conf.outfile.c_str());
//   if (file == H5I_INVALID_HID) {
//     std::cerr << "Failed to create file " << conf.outfile << std::endl;
//     return 1;
//   }
//   writeArray<2>("psi", file, c_double_id, psipdata.data(),
//                 {conf.t.n, 2 * points.size()});
//   writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g, linspace(conf.t).data(),
//                 {conf.t.n});
//   return 0;
// }

int doBasicNLin(const BasicNLinConf& conf) {
  std::vector<Point> points = readPoints(conf.pointPath);
  kdt::KDTree<Point> kdtree(points);
  f64 radius = conf.searchRadius.has_value() ? conf.searchRadius.value()
                                             : 1.01 * avgNNDist(kdtree, points);
  SparseMatrix<c64> iH =
      c64{0, -1} * SparseH(points, kdtree, radius,
                           [](Vector2d d) { return std::exp(-d.norm()); });
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  VectorXcd psi(points.size());
  for (auto& e : psi) {
    e = {cosf(dis(gen)), sinf(dis(gen))};
  }
  std::vector<c64> outdata((conf.t.n + 1) * points.size());
  memcpy(outdata.data(), psi.data(), psi.size() * sizeof(c64));
  auto rhs = basicNonLin(iH, conf.alpha);
  for (u32 i = 1; i < conf.t.n + 1; i++) {
    psi = rk4step(psi, conf.t.d(), rhs);
    memcpy(outdata.data() + i * psi.size() * sizeof(c64), psi.data(),
           psi.size() * sizeof(c64));
  }

  H5File file(conf.outfile.c_str());
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }
  writeArray<2>("data", file, c_double_id, outdata.data(),
                {conf.t.n, points.size()});
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g,
                linspace(conf.t, true).data(), {conf.t.n});
  return 0;
}

int doKuramoto(const KuramotoConf& conf) {
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
  std::ofstream fout(conf.outfile);
  fout << theta.format(oneliner) << '\n';
  auto rhs = kuramoto(conf.K, conf.N, omega);
  for (u32 i = 0; i < conf.t.n; i++) {
    theta = rk4step(theta, conf.t.d(), rhs);
    fout << theta.format(oneliner) << '\n';
  }

  fout.close();
  return 0;
}

int doExactBasic(const BasicConf& conf) {
  std::vector<Point> points = readPoints(conf.pointPath);
  kdt::KDTree<Point> kdtree(points);
  f64 radius = conf.searchRadius.has_value() ? conf.searchRadius.value()
                                             : 1.01 * avgNNDist(kdtree, points);
  auto iH = c64{0, -1} * DenseH(points, kdtree, radius,
                                [](Vector2d d) { return std::exp(-d.norm()); });
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  VectorXcd psi(points.size());
  for (auto& e : psi) {
    e = {dis(gen), dis(gen)};
  }
  std::ofstream fout(conf.outfile);
  MatrixXcd U = (iH * conf.t.d()).exp();
  fout << 0.0 << " " << psi.format(oneliner) << '\n';
  for (u32 i = 0; i < conf.t.n; i++) {
    psi = U * psi;
    fout << i * conf.t.d() << " " << psi.format(oneliner) << '\n';
  }
  fout.close();
  return 0;
}
