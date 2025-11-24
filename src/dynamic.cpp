#include "dynamic.h"
#include "SDF.h"
#include "geometry.h"
#include "unsupported/Eigen/MatrixFunctions"
#include <boost/numeric/odeint.hpp>
#include <random>
#include <toml++/toml.hpp>

using Eigen::MatrixXd, Eigen::VectorXcd;
namespace odeint = boost::numeric::odeint;

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
  SET_STRUCT_FIELD(conf, tbl, pointPath);
  SET_STRUCT_FIELD(conf, tbl, searchRadius);
  SET_STRUCT_FIELD(conf, tbl, t0);
  SET_STRUCT_FIELD(conf, tbl, t1);
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
    return iH * x + alpha * x.cwiseAbs2();
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

auto kuramoto(const SparseMatrix<f64>& K, const VectorXd& omega) {
  return [=](const VectorXd& theta) {
    MatrixXd sins = (theta * VectorXd::Ones(theta.size()).transpose() -
                     VectorXd::Ones(theta.size()) * theta.transpose())
                        .array()
                        .sin();
    return omega + (1 / (theta.size())) * (K * sins).rowwise().sum();
  };
}

auto basic(const SparseMatrix<c64>& iH) {
  return [&iH](const VectorXcd& x) { return iH * x; };
}

int doDynamic(const DynConf& conf) {
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
  std::ofstream fout("dynamic.txt");
  fout << psi.format(oneliner) << '\n';
  auto rhs = basic(iH);
  for (u32 i = 0; i < 1000; i++) {
    psi = rk4step(psi, 0.01, rhs);
    fout << psi.format(oneliner) << '\n';
  }

  fout.close();
  std::cout << "got here\n";
  return 0;
}

int doExactBasic(const DynConf& conf) {
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
  std::ofstream fout("dynamicexact.txt");
  f64 dt = 0.01;
  MatrixXcd U = (iH * dt).exp();
  fout << 0.0 << " " << psi.format(oneliner) << '\n';
  for (u32 i = 0; i < 1000; i++) {
    psi = U * psi;
    fout << i * dt << " " << psi.format(oneliner) << '\n';
  }
  fout.close();
  return 0;
}
