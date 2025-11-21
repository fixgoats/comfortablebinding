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

template <class F>
VectorXcd rk4step(const VectorXcd& x, f64 dt, F rhs) {
  VectorXcd k1 = rhs(x);
  VectorXcd k2 = rhs(x + 0.5 * dt * k1);
  VectorXcd k3 = rhs(x + 0.5 * dt * k2);
  VectorXcd k4 = rhs(x + dt * k3);
  return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
}

static Eigen::IOFormat oneliner(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                " ", " ", "", "", "", "");
int doDynamic(const DynConf& conf) {
  std::vector<Point> points = readPoints(conf.pointPath);
  kdt::KDTree<Point> kdtree(points);
  // double a = avgNNDist(kdtree, points);
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
  fout << 0.0 << " " << psi.format(oneliner) << '\n';
  auto rhs = [=](const VectorXcd& x) { return iH * x; };
  for (u32 i = 0; i < 1000; i++) {
    psi = rk4step(psi, 0.01, rhs);
    fout << 0.01 * i << " " << psi.format(oneliner) << '\n';
  }

  fout.close();
  std::cout << "got here\n";
  return 0;
}

int doExactBasic(const DynConf& conf) {
  std::vector<Point> points = readPoints(conf.pointPath);
  kdt::KDTree<Point> kdtree(points);
  double a = avgNNDist(kdtree, points);
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
