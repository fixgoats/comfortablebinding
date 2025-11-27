#include "dynamic.h"
#include "SDF.h"
#include "geometry.h"
#include "io.h"
#include "unsupported/Eigen/MatrixFunctions"
#include <boost/numeric/odeint.hpp>
#include <random>
#include <toml++/toml.hpp>

using Eigen::MatrixXd, Eigen::VectorXcd;

#define SET_STRUCT_FIELD(c, tbl, key)                                          \
  if (tbl.contains(#key))                                                      \
  c.key = *tbl[#key].value<decltype(c.key)>()

std::optional<DynConf> tomlToDynConf(const std::string& fname) {
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
    SET_STRUCT_FIELD(bc, btbl, searchRadius);
    bc.t = tblToRange(*btbl["t"].as_table());
    conf.basic = bc;
  }
  if (tbl.contains("basicNonLin")) {
    BasicNLinConf bc{};
    toml::table btbl = *tbl["basicNonLin"].as_table();
    SET_STRUCT_FIELD(bc, btbl, outfile);
    SET_STRUCT_FIELD(bc, btbl, pointPath);
    SET_STRUCT_FIELD(bc, btbl, searchRadius);
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
  return conf;
}

#undef SET_STRUCT_FIELD

// VectorXcd rhs(const VectorXcd& x, const SparseMatrix<c64>& J) { return J * x;
// }

template <class F, class State>
State rk4step(const State& x, f64 dt, F rhs) {
  State k1 = rhs(x);
  State k2 = rhs(x + 0.5 * dt * k1);
  State k3 = rhs(x + 0.5 * dt * k2);
  State k4 = rhs(x + dt * k3);
  return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
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
  Self operator*(const f64& b) const { return {b * x, b * n}; }
  Self operator+(const Self& b) const { return {x + b.x, n + b.n}; }
};

auto coupledNonLin(const SparseMatrix<c64>& iH, const VectorXd& P, f64 alpha,
                   f64 R, f64 gamma, f64 Gamma) {
  return [=](const StateAndReservoir& x) {
    return StateAndReservoir{iH * x.x + alpha * x.x.cwiseAbs2() + R * x.n -
                                 gamma * x.x,
                             P - Gamma * x.n};
  };
}

auto kuramoto(f64 K, u32 N, VectorXd omega) {
  return [=](const VectorXd& theta) {
    MatrixXd sins = (VectorXd::Ones(N) * theta.transpose() -
                     theta * VectorXd::Ones(N).transpose())
                        .array()
                        .sin();
    return VectorXd(omega + (K / N) * (sins).rowwise().sum());
  };
}

auto advancedKuramoto(f64 K, const VectorXd& omega) {
  return [=](const VectorXd& theta) {
    MatrixXd sins = (VectorXd::Ones(theta.size()) * theta.transpose() -
                     theta * VectorXd::Ones(theta.size()).transpose())
                        .array()
                        .sin();
    return VectorXd(omega +
                    (1.0 / (theta.size())) * (K * sins).rowwise().sum());
  };
}

auto basic(const SparseMatrix<c64>& iH) {
  return [&iH](const VectorXcd& x) { return VectorXcd(iH * x); };
}

int doBasic(const BasicConf& conf) {
  std::vector<Point> points = readPoints(conf.pointPath);
  kdt::KDTree<Point> kdtree(points);
  SparseMatrix<c64> iH =
      c64{0, -1} * SparseH(points, kdtree, conf.searchRadius,
                           [](Vector2d d) { return std::exp(-d.norm()); });
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  VectorXcd psi(points.size());
  for (auto& e : psi) {
    e = {dis(gen), dis(gen)};
  }
  std::ofstream fout(conf.outfile);
  fout << psi.format(oneliner) << '\n';
  auto rhs = basic(iH);
  for (u32 i = 0; i < conf.t.n; i++) {
    psi = rk4step(psi, conf.t.d(), rhs);
    fout << psi.format(oneliner) << '\n';
  }

  fout.close();
  return 0;
}

int doBasicNLin(const BasicNLinConf& conf) {
  std::vector<Point> points = readPoints(conf.pointPath);
  kdt::KDTree<Point> kdtree(points);
  SparseMatrix<c64> iH =
      c64{0, -1} * SparseH(points, kdtree, conf.searchRadius,
                           [](Vector2d d) { return std::exp(-d.norm()); });
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  VectorXcd psi(points.size());
  for (auto& e : psi) {
    e = {dis(gen), dis(gen)};
  }
  std::ofstream fout(conf.outfile);
  fout << psi.format(oneliner) << '\n';
  auto rhs = basicNonLin(iH, conf.alpha);
  for (u32 i = 0; i < conf.t.n; i++) {
    psi = rk4step(psi, conf.t.d(), rhs);
    fout << psi.format(oneliner) << '\n';
  }

  fout.close();
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
  auto iH = c64{0, -1} * DenseH(points, kdtree, conf.searchRadius,
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
