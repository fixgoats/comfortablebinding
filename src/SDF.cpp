#include "SDF.h"
#include "Eigen/Core"
#include <mdspan>
#include <iostream>

using namespace std::chrono;


void autoLimits(const VectorXd& D, RangeConf<double>& rc) {
  double max = D.maxCoeff();
  double min = D.minCoeff();
  double l = max - min;
  rc.start = min - 0.01 * l;
  rc.end = max + 0.01 * l;
}

Delta delta(const VectorXd& D, RangeConf<double> ec, double sharpening, double cutoff) {
  const double nonzero_range =
    std::sqrt(-std::log(cutoff) / sharpening);
  Delta delta(ec.n);
  for (u32 i = 0; i < ec.n; i++) {
    const double e = ec.ith(i);
    delta[i] = [&]() {
      std::vector<std::pair<double, u32>> tmp;
      tmp.reserve(5);
      for (u32 k = 0; k < D.size(); k++) {
        if (double diff = std::abs(D(k) - e); diff < nonzero_range) {
          tmp.push_back({std::exp(-sharpening * square(diff)), k});
        }
      }
      tmp.shrink_to_fit();
      return tmp;
    }();
  }
  return delta;
}

VectorXcd planeWave(Vector2d k, const std::vector<Point>& points) {
  VectorXcd tmp = VectorXcd::Zero(points.size());
  std::transform(points.begin(), points.end(), tmp.begin(), [&](Point p) {
    return (1. / sqrt(points.size())) *
           std::exp(c64{0, k(0) * p[0] + k(1) * p[1]});
  });
  return tmp.transpose();
}

std::vector<double> fullSDF(const VectorXd& D, const MatrixXcd& UH,
                            const std::vector<Point>& points, double lat_const,
                            RangeConf<double> kxc, RangeConf<double> kyc,
                            RangeConf<double>& ec, double sharpening, double cutoff, bool printProgress) {
  std::cout << "Calculating full SDF\n";
  const u32 its = kxc.n / 10;
  std::vector<double> sdf(ec.n * kyc.n * kxc.n, 0);
  auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n, ec.n);
  if (fleq(ec.start, ec.end, 1e-16)) {
    autoLimits(D, ec);
  }
  auto del = delta(D, ec, sharpening, cutoff);
  if (printProgress)
    std::cout << "[" << std::flush;
#pragma omp parallel for
  for (size_t i = 0; i < kxc.n; i++) {
    const double kx = kxc.ith(i) * 2 * M_PI / lat_const;
    for (u64 j = 0; j < kyc.n; j++) {
      const double ky =  kyc.ith(j) * 2 * M_PI / lat_const;
      const VectorXcd k_vec = UH * planeWave({kx, ky}, points);
      for (u32 k = 0; k < ec.n; k++) {
        for (const auto& pair : del[k]) {
          sdf_view[i, j, k] += pair.first * std::norm(k_vec(pair.second));
        }
      }
    }
    if (printProgress)
      if (i % its == 0)
        std::cout << "█|" << std::flush;
  }
#pragma omp barrier
  if (printProgress)
    std::cout << "█]\n";
  return sdf;
}

std::vector<double> Esection(const VectorXd& D, const MatrixXcd& UH,
                            const std::vector<Point>& points, double lat_const,
                            RangeConf<double> kxc, RangeConf<double> kyc,
                            double e, double sharpening, double cutoff, bool printProgress) {
  std::cout << "Calculating cross section of SDF at E = " << e << '\n';
  const u32 its = kxc.n / 10;
  std::vector<double> sdf(kyc.n * kxc.n, 0);
  auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
  auto del = delta(D, {e, e, 1}, sharpening, cutoff);
  if (printProgress)
    std::cout << "[" << std::flush;
  const std::vector<size_t> indices = [&](){
    std::vector<size_t> tmp;
    for (const auto pair : del[0]) {
      tmp.push_back(pair.second);
    }
    return tmp;
  }();
  MatrixXcd restrictedUH = UH(indices, Eigen::indexing::all);
#pragma omp parallel for
  for (size_t i = 0; i < kxc.n; i++) {
    const double kx = kxc.ith(i) * 2 * M_PI / lat_const;
    for (u64 j = 0; j < kyc.n; j++) {
      const double ky =  kyc.ith(j) * 2 * M_PI / lat_const;
      const VectorXcd k_vec = restrictedUH * planeWave({kx, ky}, points);
      u64 cur_element = 0;
      for (const auto& pair : del[0]) {
        sdf_view[i, j] += pair.first * std::norm(k_vec(cur_element));
        ++cur_element;
      }
    }
    if (printProgress)
      if (i % its == 0)
        std::cout << "█|" << std::flush;
  }
#pragma omp barrier
  if (printProgress)
    std::cout << "█]\n";
  return sdf;
}

std::vector<double> DOS(const VectorXd& D, const MatrixXcd& UH,
                        const std::vector<Point>& points, double lat_const, RangeConf<double> kxc,
                        RangeConf<double> kyc, RangeConf<double>& ec, double sharpening, double cutoff,
                        bool printProgress) {
  const u32 its = kxc.n / 10;
  std::vector<double> densities(ec.n, 0);
  if (fleq(ec.start, ec.end, 1e-16)) {
    autoLimits(D, ec);
  }
  auto del = delta(D, ec, sharpening, cutoff);
  if (printProgress)
    std::cout << "[" << std::flush;
  for (size_t i = 0; i < kxc.n; i++) {
    const double kx = kxc.ith(i) * 2 * M_PI / lat_const;
    for (u64 j = 0; j < kyc.n; j++) {
      const double ky = kyc.ith(j) * 2 * M_PI / lat_const;
      const VectorXcd k_vec = UH * planeWave({kx, ky}, points);
#pragma omp parallel for
      for (u32 k = 0; k < ec.n; k++) {
        for (const auto& pair : del[k]) {
          densities[k] += pair.first * std::norm(k_vec(pair.second));
        }
      }
#pragma omp barrier
    }
    if (printProgress)
      if (i % its == 0)
        std::cout << "█|" << std::flush;
  }
  if (printProgress)
    std::cout << "█]\n";
  return densities;
}

std::vector<double> disp(const VectorXd& D, const MatrixXcd& UH,
                         const std::vector<Point>& points, double lat_const,
                         RangeConf<Vector2d> kc, RangeConf<double>& ec, double sharpening, double cutoff,
                         bool printProgress) {
  std::cout << "Calculating dispersion relation\n";
  const u32 its = kc.n / 10;
  std::vector<double> disp(kc.n * ec.n, 0);
  std::cout << "Energy start value: " << ec.start << '\n';
  std::cout << "Energy end value: " << ec.start << '\n';
  if (fleq(ec.start, ec.end, 1e-16)) {
    autoLimits(D, ec);
  }
  std::cout << "Energy start value: " << ec.start << '\n';
  std::cout << "Energy end value: " << ec.end << '\n';
  auto del = delta(D, ec, sharpening, cutoff);
  auto disp_view = std::mdspan(disp.data(), ec.n, kc.n);
  if (printProgress)
    std::cout << "[" << std::flush;

  for (size_t i = 0; i < kc.n; i++) {
    const auto k = kc.ith(i) * 2 * M_PI / lat_const;
    const VectorXcd k_vec = UH * planeWave(k, points);
#pragma omp parallel for
    for (size_t j = 0; j < ec.n; j++) {
      for (const auto& pair : del[j]) {
        disp_view[j, i] += pair.first * std::norm(k_vec(pair.second));
      }
    }
#pragma omp barrier
    if (printProgress)
      if (i % its == 0)
        std::cout << "█|" << std::flush;
  }
  if (printProgress)
    std::cout << "█]\n";
  return disp;
}
MatrixXd pointsToFiniteHamiltonian(const std::vector<Point>& points,
                                    const kdt::KDTree<Point>& kdtree,
                                    f64 radius) {
  /* This function creates a hamiltonian for a simple finite lattice.
   * Can't exactly do a dispersion from this.
   */
  std::vector<Neighbour> nb_info;
  for (size_t i = 0; i < points.size(); i++) {
    auto q = points[i];
    auto nbs = kdtree.radiusSearch(q, radius);
    for (const auto idx : nbs) {
      if ((size_t)idx > i) {
        auto p = points[idx];
        Vector2d d = {p[0] - q[0], p[1] - q[1]};
        nb_info.emplace_back(i, p.idx, d);
      }
    }
  }
  return finite_hamiltonian(points.size(), nb_info, [](Vector2d) { return 1; });
}
