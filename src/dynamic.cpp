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
    if (btbl.contains("searchRadius")) {
      bc.searchRadius = btbl["searchRadius"].value<f64>().value();
    }
    bc.t = tblToRange(*btbl["t"].as_table());
    conf.basic = bc;
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
  std::ofstream fout(conf.outfile);
  auto rhs = basic(iH);

  std::vector<c64> orderparam(conf.t.n);
  for (u32 i = 0; i < conf.t.n; i++) {
    psi = rk4step(psi, conf.t.d(), rhs);
    orderparam[i] =
        (c64{0, 1} * psi.array().arg()).exp().sum() / (f64)points.size();
    // memcpy(outdata.data() + i * points.size(), psi.data(), points.size());
  }
  // H5File file(conf.outfile.c_str());
  hid_t file =
      H5Fcreate(conf.outfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
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
    memcpy(outdata.data() + i * points.size(), psi.data(), points.size());
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
