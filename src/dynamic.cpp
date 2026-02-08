#include "dynamic.h"
#include "SDF.h"
#include "geometry.h"
#include "io.h"
#include "logging.hpp"
#include "unsupported/Eigen/MatrixFunctions"
// #include "vkcore.hpp"
#include <print>
#include <random>
#include <toml++/toml.hpp>

using Eigen::MatrixXd, Eigen::VectorXcd, Eigen::MatrixX2cd;

#define SET_STRUCT_FIELD(c, tbl, key)                                          \
  if (tbl.contains(#key))                                                      \
  c.key = *tbl[#key].value<decltype(c.key)>()

std::optional<DynConf> tomlToDynConf(const std::string& fname) {
  std::print("afhjsd {}", 3 * 4);
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
  if (tbl.contains("tetm")) {
    TETMConf tc;
    toml::table ttbl = *tbl["tetm"].as_table();
    SET_STRUCT_FIELD(tc, ttbl, outfile);
    SET_STRUCT_FIELD(tc, ttbl, pointPath);
    SET_STRUCT_FIELD(tc, ttbl, alpha);
    SET_STRUCT_FIELD(tc, ttbl, p);
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
    if (btbl.contains("searchRadius")) {
      bc.searchRadius = btbl["searchRadius"].value<f64>().value();
    }
    SET_STRUCT_FIELD(bc, btbl, alpha);
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

struct TETM {
  typedef TETM Self;

  VectorXcd psip;
  VectorXcd psim;

  TETM(u64 n) : psip{n}, psim{n} {}
  TETM(VectorXcd psip, VectorXcd psim) : psip{psip}, psim{psip} {}
  size_t byteSize() { return 2 * psip.size() * sizeof(c64); }
  Self& operator*=(f64 b) {
    psip *= b;
    psim *= b;
    return *this;
  }
  friend Self operator*(Self lhs, f64 b) { return lhs *= b; }
  Self& operator+=(const Self& rhs) {
    psip += rhs.psip;
    psim += rhs.psim;
    return *this;
  }
  friend Self operator+(Self lhs, const Self& b) { return lhs += b; }
  Self& operator-=(const Self& rhs) {
    psip -= rhs.psip;
    psim -= rhs.psim;
    return *this;
  }
  friend Self operator-(Self lhs, const Self& rhs) { return lhs -= rhs; }
};

auto tetmNonLin(const SparseMatrix<c64>& iJ, const SparseMatrix<c64>& iL, f64 p,
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

MatrixX2cd explTETM(MatrixX2cd psi, const SparseMatrix<c64>& iJ,
                    const SparseMatrix<c64>& iL, f64 p, f64 alpha) {
  MatrixX2cd first = (p - 1) * psi;
  MatrixX2cd second = -c64{0, alpha} * psi.cwiseAbs2().cwiseProduct(psi);
  MatrixX2cd third = iJ * psi;
  MatrixX2cd fourth(psi.rows(), 2);
  return first + second + third + fourth;
}

MatrixX2cd tetmRK4Step(MatrixX2cd psi, const SparseMatrix<c64>& iJ,
                       const SparseMatrix<c64>& iL, f64 p, f64 alpha, f64 dt) {

  MatrixX2cd k1 = explTETM(psi, iJ, iL, p, alpha);
  MatrixX2cd k2 = explTETM(psi + 0.5 * dt * k1, iJ, iL, p, alpha);
  MatrixX2cd k3 = explTETM(psi + 0.5 * dt * k2, iJ, iL, p, alpha);
  MatrixX2cd k4 = explTETM(psi + dt * k3, iJ, iL, p, alpha);
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

  std::vector<c64> orderparam(conf.t.n);
  for (u32 i = 0; i < conf.t.n; i++) {
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
  writeArray<1>("data", file, c_double_id, orderparam.data(), {conf.t.n});
  std::cout << "Writing corresponding times\n";
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g, linspace(conf.t).data(),
                {conf.t.n});

  return 0;
}

int doTETM(const TETMConf& conf) {
  logDebug("Function: doTETM.");
  std::vector<Point> points = readPoints(conf.pointPath);
  logDebug("Read points.");
  kdt::KDTree<Point> kdtree(points);
  logDebug("Built kdtree");
  f64 radius = conf.searchRadius.has_value() ? conf.searchRadius.value()
                                             : 1.01 * avgNNDist(kdtree, points);
  logDebug("Using search radius: {}", radius);
  SparseMatrix<c64> iJ = SparseHC(points, kdtree, radius, [](Vector2d d) {
    return c64{0, -1} * std::exp(-d.norm());
  });
  logDebug("Made sparse matrix iJ.");
  SparseMatrix<c64> iL =
      c64{0, -1} * SparseHC(points, kdtree, radius, [](Vector2d d) {
        return 0.5 * std::exp(-d.norm()) *
               c64{1 - 2 * d(1) * d(1) / d.squaredNorm(),
                   2 * d(0) * d(1) / d.squaredNorm()};
      });
  logDebug("Made sparse matrix iL.");
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.0, 2 * M_PI);
  logDebug("Allocating psi.");
  MatrixX2cd psi(points.size(), 2);
  u32 psin = psi.rows();
  u32 psim = psi.cols();
  logDebug("Allocated psi with dims {}x{}", psin, psim);
  logDebug("Writing random coordinates to psi.");
  for (u64 i = 0; i < psi.size(); i++) {
    *(psi.data() + i) = {cos(dis(gen)), sin(dis(gen))};
  }
  u32 overallSize = conf.t.n * points.size();
  logDebug("Allocating psipdata with {} elements.", overallSize);
  std::vector<c64> psidata(conf.t.n * points.size() * 2);
  // logDebug("Allocating psimdata with {} elements.", overallSize);
  // std::vector<c64> psimdata(conf.t.n * points.size());
  logDebug("Testing allocation first");
  MatrixX2cd first = (conf.p - 1) * psi;
  logDebug("Testing allocation second");
  MatrixX2cd second = -c64{0, conf.alpha} * psi.cwiseAbs2().transpose() * psi;
  logDebug("Testing allocation third");
  MatrixX2cd third = iJ * psi;
  logDebug("Testing allocation fourth");
  MatrixX2cd fourth(psi.rows(), 2);
  logDebug("Testing spm v multiplication");
  MatrixXcd b = iL * psi.col(1);
  logDebug("Setting other fourth values");
  fourth.col(1) = -iL.conjugate() * psi.col(0);
  // auto rhs = tetmNonLin(iJ, iL, conf.p, conf.alpha);
  logDebug("Created rhs function.");
  size_t byteSize = psi.size() * sizeof(c64);
  for (u32 i = 0; i < conf.t.n; i++) {
    psi = tetmRK4Step(psi, iJ, iL, conf.p, conf.alpha, conf.t.d());
    u32 n = psi.rows();
    u32 m = psi.cols();
    logDebug("Return psiish with dims {}x{}", n, m);
    logDebug("Copying to psipdata.");
    memcpy(psidata.data() + i * byteSize, psi.data(), byteSize);
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
                {conf.t.n, points.size(), 2});
  // logDebug("Writing psim to file.");
  // writeArray<2>("psip", file, c_double_id, psipdata.data(),
  //               {conf.t.n, points.size()});
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g, linspace(conf.t).data(),
                {conf.t.n});
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
  std::vector<c64> outdata(conf.t.n * points.size());
  auto rhs = basicNonLin(iH, conf.alpha);
  for (u32 i = 0; i < conf.t.n; i++) {
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
  writeArray<1>("time", file, H5T_NATIVE_DOUBLE_g, linspace(conf.t).data(),
                {conf.t.n});
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
