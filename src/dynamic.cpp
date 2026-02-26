#include "dynamic.h"
#include "SDF.h"
#include "geometry.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include "io.h"
#include "logging.hpp"
#include "unsupported/Eigen/MatrixFunctions"
#include <gsl/gsl_sf.h>
// #include "vkcore.hpp"
#include <print>
#include <random>
#include <toml++/toml.hpp>

using Eigen::MatrixXd, Eigen::VectorXcd, Eigen::MatrixX2cd;

#define SET_STRUCT_FIELD(c, tbl, key)                                          \
  if (tbl.contains(#key))                                                      \
  c.key = *tbl[#key].value<decltype(c.key)>()

std::optional<DynConf> tomlToDynConf(const std::string& fname) {
  logDebug("Function: tomlToDynConf.");
  toml::table tbl;
  try {
    tbl = toml::parse_file(fname);
  } catch (const std::exception& err) {
    std::cerr << "Parsing file " << fname
              << " failed with exception: " << err.what() << '\n';
    return {};
  }
  DynConf conf{};
  if (tbl.contains("basic")) {
    logDebug("Writing config for basic.");
    BasicConf bc{};
    toml::table btbl = *tbl["basic"].as_table();
    SET_STRUCT_FIELD(bc, btbl, outfile);
    SET_STRUCT_FIELD(bc, btbl, pointPath);
    if (btbl.contains("searchRadius")) {
      bc.searchRadius = btbl["searchRadius"].value<f64>().value();
    }
    bc.t = tblToRange(*btbl["t"].as_table());
    conf.basic = bc;
  }
  if (tbl.contains("basicdistance")) {
    logDebug("Writing config for distance scan");
    BasicDistanceConf bc{};
    toml::table btbl = *tbl["basicdistance"].as_table();
    SET_STRUCT_FIELD(bc, btbl, outfile);
    SET_STRUCT_FIELD(bc, btbl, alpha);
    SET_STRUCT_FIELD(bc, btbl, p);
    SET_STRUCT_FIELD(bc, btbl, j);
    bc.t = tblToRange(*btbl["t"].as_table());
    bc.sep = tblToRange(*btbl["sep"].as_table());
    conf.bd = bc;
  }
  if (tbl.contains("tetm")) {
    TETMConf tc;
    toml::table ttbl = *tbl["tetm"].as_table();
    SET_STRUCT_FIELD(tc, ttbl, outfile);
    SET_STRUCT_FIELD(tc, ttbl, pointPath);
    SET_STRUCT_FIELD(tc, ttbl, alpha);
    SET_STRUCT_FIELD(tc, ttbl, p);
    SET_STRUCT_FIELD(tc, ttbl, j);
    SET_STRUCT_FIELD(tc, ttbl, rscale);
    if (ttbl.contains("searchRadius")) {
      tc.searchRadius = ttbl["searchRadius"].value<f64>().value();
    }
    tc.t = tblToRange(*ttbl["t"].as_table());

    conf.tetm = tc;
  }
  if (tbl.contains("delay")) {
    TETMConf tc;
    toml::table ttbl = *tbl["delay"].as_table();
    SET_STRUCT_FIELD(tc, ttbl, outfile);
    SET_STRUCT_FIELD(tc, ttbl, pointPath);
    SET_STRUCT_FIELD(tc, ttbl, alpha);
    SET_STRUCT_FIELD(tc, ttbl, p);
    SET_STRUCT_FIELD(tc, ttbl, j);
    if (ttbl.contains("searchRadius")) {
      tc.searchRadius = ttbl["searchRadius"].value<f64>().value();
    }
    tc.t = tblToRange(*ttbl["t"].as_table());

    conf.tetm = tc;
  }
  if (tbl.contains("basicNonLin")) {
    BasicNLinConf bc{};
    toml::table btbl = *tbl["basicNonLin"].as_table();
    SET_STRUCT_FIELD(bc, btbl, outfile);
    SET_STRUCT_FIELD(bc, btbl, pointPath);
    SET_STRUCT_FIELD(bc, btbl, alpha);
    if (btbl.contains("searchRadius")) {
      bc.searchRadius = btbl["searchRadius"].value<f64>().value();
    }
    bc.t = tblToRange(*btbl["t"].as_table());
    conf.basicnlin = bc;
  }
  if (tbl.contains("kuramoto")) {
    KuramotoConf kc{};
    auto ktbl = *tbl["kuramoto"].as_table();
    SET_STRUCT_FIELD(kc, ktbl, outfile);
    SET_STRUCT_FIELD(kc, ktbl, K);
    SET_STRUCT_FIELD(kc, ktbl, N);
    kc.t = tblToRange(*ktbl["t"].as_table());
    conf.kuramoto = kc;
  }
  logDebug("Exiting tomlToDynConf.");
  return conf;
}

#undef SET_STRUCT_FIELD

// VectorXcd rhs(const VectorXcd& x, const SparseMatrix<c64>& J) { return J * x;
// }

template <class F, class State>
State rk4step(const State& x, f64 dt, F rhs) {
  logDebug("Function: rk4step.");
  State k1 = rhs(x);
  State k2 = rhs(x + 0.5 * dt * k1);
  State k3 = rhs(x + 0.5 * dt * k2);
  State k4 = rhs(x + dt * k3);
  State ret = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
  logDebug("Exiting rk4step.");
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
    logDebug("Function: tetmNonLin.");
    logDebug("Making first matrix");
    MatrixX2cd first = (p - 1) * psi;
    u32 n = first.rows();
    u32 m = first.cols();
    logDebug("First matrix has dims {}x{}", n, m);
    logDebug("Making second matrix");
    MatrixX2cd second = -c64{0, alpha} * psi.cwiseAbs2().cwiseProduct(psi);
    n = second.rows();
    m = second.cols();
    logDebug("Second matrix has dims {}x{}", n, m);
    logDebug("Making third matrix");
    MatrixX2cd third = iJ * psi;
    n = third.rows();
    m = third.cols();
    logDebug("Third matrix has dims {}x{}", n, m);
    logDebug("Making fourth matrix");
    MatrixX2cd fourth(psi.rows(), 2);
    fourth.col(0) = iL * psi.col(1);
    fourth.col(1) = -iL.conjugate() * psi.col(0);
    n = fourth.rows();
    m = fourth.cols();
    logDebug("Fourth matrix has dims {}x{}", n, m);
    logDebug("Exiting tetmNonLin.");
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
  logDebug("Function: doDistanceScan");
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  const auto seps = linspace(conf.sep, true);
  const Eigen::VectorXd times = linspace(conf.t, true);
  std::vector<c64> psil(conf.t.n);
  std::vector<c64> psir(conf.t.n);
  logDebug("Entering separation loop.");
  const auto rhs = noCoupling(conf.p, conf.alpha);
  const f64 seed1 = dis(gen);
  const f64 seed2 = dis(gen);
  Eigen::Vector2cd psi{c64{cos(seed1), sin(seed1)},
                       c64{cos(seed2), sin(seed2)}};
  psil[0] = psi.x();
  psir[0] = psi.y();
  logDebug("Entering solution loop.");
  for (u64 j = 1; j < conf.t.n; ++j) {
    psi = rk4step(psi, conf.t.d(), rhs);
    psil[j] = psi.x();
    psir[j] = psi.y();
  }
  logDebug("Finished calculating.");

  H5File file(conf.outfile.c_str());
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }
  logDebug("Writing psil to file.");
  writeArray<1>("psil", file, c_double_id, psil.data(), {conf.t.n});
  logDebug("Writing psir to file.");
  writeArray<1>("psir", file, c_double_id, psir.data(), {conf.t.n});
  logDebug("Writing time to file.");
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g, (void*)times.data(),
                {conf.t.n});
  logDebug("Exiting doNoCoupling.");
  return 0;
}

int doNCDD(const BasicDistanceConf& conf) {
  logDebug("Function: doDistanceScan");
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  const auto seps = linspace(conf.sep, true);
  const Eigen::VectorXd times = linspace(conf.t, true);
  std::vector<c64> psil(conf.t.n);
  std::vector<c64> psir(conf.t.n);
  logDebug("Entering separation loop.");
  const auto rhs = noCouplingDrivenDissipative(conf.p, conf.alpha);
  const f64 seed1 = dis(gen);
  const f64 seed2 = dis(gen);
  Eigen::Vector2cd psi{c64{cos(seed1), sin(seed1)},
                       c64{cos(seed2), sin(seed2)}};
  psil[0] = psi.x();
  psir[0] = psi.y();
  logDebug("Entering solution loop.");
  for (u64 j = 1; j < conf.t.n; ++j) {
    psi = rk4step(psi, conf.t.d(), rhs);
    psil[j] = psi.x();
    psir[j] = psi.y();
  }
  logDebug("Finished calculating.");

  H5File file(conf.outfile.c_str());
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }
  logDebug("Writing psil to file.");
  writeArray<1>("psil", file, c_double_id, psil.data(), {conf.t.n});
  logDebug("Writing psir to file.");
  writeArray<1>("psir", file, c_double_id, psir.data(), {conf.t.n});
  logDebug("Writing time to file.");
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g, (void*)times.data(),
                {conf.t.n});
  logDebug("Exiting doDistanceScan.");
  return 0;
}

int doDistanceScan(const BasicDistanceConf& conf) {
  logDebug("Function: doDistanceScan");
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  const auto seps = linspace(conf.sep, true);
  const Eigen::VectorXd times = linspace(conf.t, true);
  std::vector<c64> psil(conf.t.n * conf.sep.n);
  std::vector<c64> psir(conf.t.n * conf.sep.n);
  logDebug("Entering separation loop.");
#pragma omp parallel for
  for (u64 i = 0; i < conf.sep.n; ++i) {
    const auto rhs = basicHankel(conf.p, conf.alpha, conf.j, conf.sep.ith(i));
    const f64 seed1 = dis(gen);
    const f64 seed2 = dis(gen);
    Eigen::Vector2cd psi{c64{cos(seed1), sin(seed1)},
                         c64{cos(seed2), sin(seed2)}};
    psil[i * conf.t.n] = psi.x();
    psir[i * conf.t.n] = psi.y();
    logDebug("Entering solution loop.");
    for (u64 j = 1; j < conf.t.n; ++j) {
      psi = rk4step(psi, conf.t.d(), rhs);
      psil[i * conf.t.n + j] = psi.x();
      psir[i * conf.t.n + j] = psi.y();
    }
  }
  logDebug("Finished calculating.");

  H5File file(conf.outfile.c_str());
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }
  logDebug("Writing psil to file.");
  writeArray<2>("psil", file, c_double_id, psil.data(), {conf.sep.n, conf.t.n});
  logDebug("Writing psir to file.");
  writeArray<2>("psir", file, c_double_id, psir.data(), {conf.sep.n, conf.t.n});
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g, (void*)times.data(),
                {conf.t.n});
  logDebug("Exiting doDistanceScan.");
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
  logDebug("Function: doBasicHankelDD.");
  HighFive::File pc(conf.pointPath, HighFive::File::ReadOnly);
  logDebug("Read pointfile.");
  Eigen::MatrixX2d points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
  logDebug("Read points.");
  Eigen::MatrixX2i couplings =
      pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
  logDebug("Read couplings.");
  /*const f64 radius = conf.searchRadius.has_value() ? conf.searchRadius.value()
                                                   : 1.01 * avgradius;
  std::cout << "Radius is: " << radius << '\n';
  logDebug("Using search radius: {}", radius);
  std::vector<Neighbour> toadie = pointsToNbs(points, kdtree, radius);*/
  const f64 rscale = conf.rscale;
  const SparseMatrix<c64> J =
      conf.j * SparseC(points, couplings, [rscale](Vector2d d) {
        return c64{gsl_sf_bessel_J0(rscale * d.norm()),
                   gsl_sf_bessel_Y0(rscale * d.norm())};
      });
  logDebug("Made sparse matrix J.");
  auto sol = Eigen::ComplexEigenSolver<MatrixXcd>(MatrixXcd(J));
  auto max_coeff = std::ranges::max_element(
      sol.eigenvalues(), [](c64 a, c64 b) { return a.real() < b.real(); });
  std::cout << "Highest eigenvalue is: " << max_coeff->real() << '\n';
  const f64 p = (conf.p - 1) * max_coeff->real();
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  logDebug("Allocating psi.");
  VectorXcd psi(points.rows());
  u64 n = psi.rows();
  u64 m = psi.cols();
  logDebug("Allocated psi with dims {}x{}", n, m);
  logDebug("Writing random coordinates to psi.");
  for (u64 i = 0; i < psi.size(); i++) {
    auto x = dis(gen);
    *(psi.data() + i) = {1e-4 * cos(x), 1e-4 * sin(x)};
  }
  const u32 overallSize = conf.t.n * psi.size();
  logDebug("Allocating psipdata with {} elements.", overallSize);
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
    logDebug("Return psiish with dims {}x{}", n, m);
    logDebug("Copying to psipdata.");
    psidata(Eigen::indexing::all, i) = psi;
    // memcpy(psidata.data() + i * psi.size(), psi.data(), byteSize);
    // logDebug("Copying to psimdata.");
    // memcpy(psimdata.data() + i * byteSize, psi.data(), byteSize);
  }
  logDebug("Finished calculating.");

  /*H5File file(conf.outfile.c_str());
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }*/
  HighFive::File file(conf.outfile, HighFive::File::Truncate);
  logDebug("Writing psip to file.");

  /*writeArray<2>("psi", file, c_double_id, psidata.data(),
                {conf.t.n, points.size()});*/
  file.createDataSet("psi", psidata);
  file.createDataSet("points", points);
  file.createDataSet("couplings", couplings);
  file.createDataSet("time", linspace(conf.t, true));
  // paramType.commit(file, "param_type");
  HighFive::DataSet paramSet = file.createDataSet(
      "params", Params{conf.p, conf.alpha, conf.j, conf.rscale});
  /*f64 params[4] = ;
  paramSet.write(params);
  file.flush();*/

  /*hsize_t point_mem_dims[2] = {points.size(), 3};
  hid_t point_mem_space = H5Screate_simple(2, point_mem_dims, nullptr);
  hsize_t point_file_dims[2] = {points.size(), 2};
  hid_t point_file_space = H5Screate_simple(2, point_file_dims, nullptr);
  hsize_t start[2] = {0, 0};
  hsize_t stride[2] = {1, 1};
  hsize_t count[2] = {points.size(), 1};
  hsize_t block[2] = {1, 2};
  H5Sselect_hyperslab(point_mem_space, H5S_SELECT_SET, start, stride, count,
                      block);
  hid_t pointSet =
      H5Dcreate2(file, "points", H5T_NATIVE_DOUBLE, point_file_space,
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(pointSet, H5T_NATIVE_DOUBLE, point_mem_space, point_file_space,
           H5P_DEFAULT, points.data());
  hsize_t nb_mem_dims[2] = {couplings.size(), 4};
  hid_t nb_mem_space = H5Screate_simple(2, nb_mem_dims, nullptr);
  hsize_t nb_file_dims[2] = {toadie.size(), 2};
  hid_t nb_file_space = H5Screate_simple(2, nb_file_dims, nullptr);
  count[0] = toadie.size();
  H5Sselect_hyperslab(nb_mem_space, H5S_SELECT_SET, start, stride, count,
                      block);
  hid_t nbSet = H5Dcreate2(file, "couplings", H5T_NATIVE_INT64, nb_file_space,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(nbSet, H5T_NATIVE_INT64, nb_mem_space, nb_file_space, H5P_DEFAULT,
           toadie.data());
  hid_t paramType = H5Tcreate(H5T_COMPOUND, 8 * 5);
  H5Tinsert(paramType, "p", 0, H5T_NATIVE_DOUBLE);
  H5Tinsert(paramType, "alpha", 8, H5T_NATIVE_DOUBLE);
  H5Tinsert(paramType, "j", 16, H5T_NATIVE_DOUBLE);
  H5Tinsert(paramType, "rscale", 24, H5T_NATIVE_DOUBLE);
  H5Tinsert(paramType, "radius", 32, H5T_NATIVE_DOUBLE);
  hsize_t paramDim = 1;
  hid_t paramSpace = H5Screate_simple(1, &paramDim, nullptr);
  hid_t paramSet = H5Dcreate2(file, "params", paramType, paramSpace,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  f64 params[5] = {conf.p, conf.alpha, conf.j, conf.rscale, radius};
  H5Dwrite(paramSet, paramType, paramSpace, paramSpace, H5P_DEFAULT, params);

  // logDebug("Writing psim to file.");
  // writeArray<2>("psip", file, c_double_id, psipdata.data(),
  //               {conf.t.n, points.size()});
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE, linspace(conf.t, true).data(),
                {conf.t.n});*/
  logDebug("Exiting doTETM.");
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
  logDebug("Function: doTETM.");
  const std::vector<Point> points = readPoints(conf.pointPath);
  logDebug("Read points.");
  kdt::KDTree<Point> kdtree(points);
  logDebug("Built kdtree");
  const f64 avgradius = avgNNDist(kdtree, points);
  const f64 radius = conf.searchRadius.has_value() ? conf.searchRadius.value()
                                                   : 1.01 * avgradius;
  std::cout << "Radius is: " << radius << '\n';
  logDebug("Using search radius: {}", radius);
  SparseMatrix<c64> J =
      conf.j * SparseHC(points, kdtree, radius, [](Vector2d d) {
        return c64{gsl_sf_bessel_J0(d.norm()), gsl_sf_bessel_Y0(d.norm())};
      });
  logDebug("Made sparse matrix iJ.");
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
  logDebug("Made sparse matrix iL.");
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  logDebug("Allocating psi.");
  MatrixX2cd psi(points.size(), 2);
  u64 n = psi.rows();
  u64 m = psi.cols();
  logDebug("Allocated psi with dims {}x{}", n, m);
  logDebug("Writing random coordinates to psi.");
  for (u64 i = 0; i < psi.size(); i++) {
    auto x = dis(gen);
    *(psi.data() + i) = {cos(x), sin(x)};
  }
  u32 overallSize = conf.t.n * points.size();
  logDebug("Allocating psipdata with {} elements.", overallSize);
  std::vector<c64> psidata((conf.t.n) * points.size() * 2);
  size_t byteSize = psi.size() * sizeof(c64);
  memcpy(psidata.data(), psi.data(), byteSize);
  // logDebug("Allocating psimdata with {} elements.", overallSize);
  // std::vector<c64> psimdata(conf.t.n * points.size());
  // logDebug("Testing allocation first");
  // MatrixX2cd first = (conf.p - 1) * psi;
  // logDebug("Testing allocation second");
  // MatrixX2cd second = -c64{0, conf.alpha} * psi.cwiseAbs2().transpose() *
  // psi; logDebug("Testing allocation third"); MatrixX2cd third = iJ * psi;
  // logDebug("Testing allocation fourth");
  // MatrixX2cd fourth(psi.rows(), 2);
  // logDebug("Testing spm v multiplication");
  // MatrixXcd b = iL * psi.col(1);
  // logDebug("Setting other fourth values");
  // fourth.col(1) = -iL.conjugate() * psi.col(0);
  // auto rhs = tetmNonLin(iJ, iL, conf.p, conf.alpha);
  // logDebug("Created rhs function.");
  std::cout << "byteSize is: " << byteSize << '\n';
  std::cout << "Byte size of psidata is: " << psidata.size() * sizeof(c64);
  for (u32 i = 1; i < conf.t.n; i++) {
    psi = tetmRK4Step(psi, J, L, conf.p, conf.alpha, conf.t.d());
    u64 n = psi.rows();
    u64 m = psi.cols();
    logDebug("Return psiish with dims {}x{}", n, m);
    logDebug("Copying to psipdata.");
    memcpy(psidata.data() + i * psi.size(), psi.data(), byteSize);
    // logDebug("Copying to psimdata.");
    // memcpy(psimdata.data() + i * byteSize, psi.data(), byteSize);
  }
  logDebug("Finished calculating.");

  H5File file(conf.outfile.c_str());
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.outfile << std::endl;
    return 1;
  }
  logDebug("Writing psip to file.");
  writeArray<3>("psi", file, c_double_id, psidata.data(),
                {conf.t.n, 2, points.size()});
  // logDebug("Writing psim to file.");
  // writeArray<2>("psip", file, c_double_id, psipdata.data(),
  //               {conf.t.n, points.size()});
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g,
                linspace(conf.t, true).data(), {conf.t.n});
  logDebug("Exiting doTETM.");
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
//                                          2 * d(0) * d(1) / d.squaredNorm()};
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
