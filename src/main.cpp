#include "Eigen/Dense"
#include "H5Cpp.h"
#include "betterexc.h"
#include "kdtree.h"
#include "lodepng.h"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "typedefs.h"
#include <H5public.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cxxopts.hpp>
#include <exception>
#include <iostream>
#include <iterator>
#include <mdspan>
#include <random>
#include <ranges>
#include <sstream>
#include <toml++/toml.hpp>

using Eigen::MatrixXd, Eigen::MatrixXcd, Eigen::VectorXd, Eigen::MatrixX2d,
    Eigen::VectorXcd, Eigen::Vector2d, Eigen::SelfAdjointEigenSolver,
    Eigen::VectorXi;
// Along with gauss_sharpening, controls acceptable deviation from each e for an
// eigenvalue to be considered. Increase for less acceptable deviation.
constexpr double gauss_cutoff = 0.1;
// increase to decrease fuzziness. If too sharp with too high cutoff, could miss
// some or all structure.
constexpr double gauss_sharpening = 200;
// The acceptable deviation from energy value
static const double nonzero_range =
    std::sqrt(std::log(std::pow(gauss_cutoff, -gauss_sharpening)));

static Eigen::IOFormat defaultFormat(Eigen::StreamPrecision,
                                     Eigen::DontAlignCols, " ", "\n", "", "",
                                     "", "");

extern "C" {
extern int zheevr_(const char* JOBZ, const char* RANGE, const char* UPLO,
                   int* N, double* A, int* LDA, double* VL, double* VU, int* IL,
                   int* IU, double* ABSTOL, int* M, double* W, double* Z,
                   int* LDZ, int* ISUPPZ, double* WORK, int* LWORK,
                   double* RWORK, int* LRWORK, int* IWORK, int* LIWORK,
                   int* INFO);
extern double dlamch_(char* cmach);
}

int writeh5wexc(hid_t dataset, hid_t type, hid_t fspace, const void* data) {
    return H5Dwrite(dataset, type, H5S_ALL, fspace, H5P_DEFAULT, data);
}

template <class Func>
MatrixXcd couplingmat(const VectorXd& xs, const VectorXd& ys, Vector2d k,
                      f64 rsq0, Func f) {
  const u32 n = xs.size();
  MatrixXcd J = MatrixXcd::Zero(n, n);
  for (u32 j = 0; j < n - 1; j++) {
    for (u32 i = j + 1; i < n; i++) {
      Vector2d r{xs[i] - xs[j], ys[i] - ys[j]};
      if (r.squaredNorm() < rsq0) {
        c64 val = f(r.norm()) * std::exp(c64{0, k.dot(r)});
        J(i, j) += val;
        J(i, j) += std::conj(val);
      }
    }
  }
  return J;
}

template <class Func>
MatrixXd couplingmat(const MatrixX2d& points, f64 rsq, Func f) {
  const u32 n = points.rows();
  MatrixXd J(n, n);
  for (u32 j = 0; j < n - 1; j++) {
    for (u32 i = j + 1; i < n; i++) {
      const Eigen::Vector2d x = points(i, {0, 1}) - points(j, {0, 1});
      if (x.squaredNorm() < rsq) {
        J(j, i) = f(x.norm());
        J(i, j) = f(x.norm());
      } else {
        J(j, i) = 0;
        J(i, j) = 0;
      }
    }
  }
  return J;
}

struct Point : std::array<double, 2> {
  static constexpr int DIM = 2;
  u32 idx;

  Point() {}
  Point(double x, double y, u32 idx) : idx{idx} {
    (*this)[0] = x;
    (*this)[1] = y;
  }

  double sqdist(const Point& p) const {
    return square((*this)[0] - p[0]) + square((*this)[1] - p[1]);
  }

  double dist(const Point& p) const {
    return sqrt(square((*this)[0] - p[0]) + square((*this)[1] - p[1]));
  }

  friend std::ostream& operator<<(std::ostream& os, const Point& pt) {
    return os << std::format("({}, {}, {})", pt[0], pt[1], pt.idx);
  }
};

struct Line {
  friend std::istream& operator>>(std::istream& is, Line& line) {
    return std::getline(is, line.lineTemp);
  }

  // Output function.
  friend std::ostream& operator<<(std::ostream& os, const Line& line) {
    return os << line.lineTemp;
  }

  // cast to needed result
  operator std::string() const { return lineTemp; }
  // Temporary Local storage for line
  std::string lineTemp{};
};

std::vector<Point> readPoints(const std::string& fname) {
  u32 m = 0;
  std::ifstream f(fname);
  std::vector<std::string> allLines{std::istream_iterator<Line>(f),
                                    std::istream_iterator<Line>()};
  m = allLines.size();
  // f.clear();
  // f.seekg(0, std::ios::beg);
  std::vector<Point> M(m);
  for (u32 j = 0; j < m; j++) {
    std::istringstream stream(allLines[j]);
    std::vector<double> v{std::istream_iterator<double>(stream),
                          std::istream_iterator<double>()};
    M[j] = {v[0], v[1], j};
  }
  return M;
}

template <class D>
void saveEigen(const std::string& fname, const Eigen::MatrixBase<D>& x) {
  std::ofstream f(fname);
  f << x.format(defaultFormat);
  f.close();
}

template <class T>
Eigen::MatrixX<T> readEigenStream(const std::string& fname) {
  std::string line;
  std::ifstream f(fname);
  u32 m = 0;
  u32 n = 0;
  while (std::getline(f, line)) {
    if (n == 0) {
      auto splits = line | std::ranges::views::split(' ');
      for (const auto& _ : splits) {
        m += 1;
      }
    }
    n += 1;
  }
  f.clear();
  f.seekg(0, std::ios::beg);
  Eigen::MatrixX<T> M(n, m);
  u32 j = 0;
  u32 i = 0;
  M(i, j) >> f;
}

template <class T>
Eigen::MatrixX<T> readEigen(std::string fname) {
  u32 m = 0;
  u32 n = 0;
  std::ifstream f(fname);
  std::vector<std::string> allLines{std::istream_iterator<Line>(f),
                                    std::istream_iterator<Line>()};
  m = allLines.size();
  std::istringstream s(allLines[0]);
  std::vector<T> v{std::istream_iterator<T>(s), std::istream_iterator<T>()};
  n = v.size();
  // f.clear();
  // f.seekg(0, std::ios::beg);
  Eigen::MatrixX<T> M(m, n);
  u32 j = 0;
  for (const auto& line : allLines) {
    std::istringstream stream(line);
    std::vector<T> v{std::istream_iterator<T>(stream),
                     std::istream_iterator<T>()};
    u32 i = 0;
    for (const auto bleh : v) {
      M(j, i) = bleh;
      i += 1;
    }
    j += 1;
  }
  return M;
}

template <class PointT>
double max_norm_dist(const PointT& q, const PointT& p) {
  double max = 0;
  for (size_t i = 0; i < PointT::DIM; i++) {
    double d = abs(q[i] - p[i]);
    if (d > max)
      max = d;
  }
  return max;
}

struct Neighbour {
  size_t i;
  size_t j;
  Vector2d d;

  friend std::ostream& operator<<(std::ostream& os, const Neighbour& nb) {
    return os << '(' << nb.i << ", " << nb.j << ", " << nb.d << ')';
  }
};

template <class Func>
void update_hamiltonian(MatrixXcd& H, const std::vector<Neighbour>& nbs,
                        Vector2d k, Func f, bool reset = false) {
  if (reset) {
    for (const auto& nb : nbs) {
      H(nb.i, nb.j) = 0;
      H(nb.j, nb.i) = 0;
    }
  }
  for (const auto& nb : nbs) {
    c64 val = f(nb.d) * std::exp(c64{0, k.dot(nb.d)});
    H(nb.i, nb.j) += val;
    H(nb.j, nb.i) += std::conj(val);
  }
}

template <class Func>
MatrixXcd finite_hamiltonian(u32 n_points, const std::vector<Neighbour>& nbs,
                             Func f) {
  MatrixXcd H = MatrixXcd::Zero(n_points, n_points);
  for (const auto& nb : nbs) {
    c64 val = f(nb.d);
    H(nb.i, nb.j) = val;
    H(nb.j, nb.i) = std::conj(val);
  }
  return H;
}

template <class PointT>
double avgNNDist(kdt::KDTree<PointT>& kdtree,
                 const std::vector<PointT>& points) {
  std::vector<double> nn_dist(points.size());
  for (size_t i = 0; i < points.size(); i++) {
    int idx = kdtree.knnSearch(points[i], 2)[1];
    nn_dist[i] = points[i].dist(points[idx]);
  }
  double totalnndist =
      std::accumulate(nn_dist.cbegin() + 1, nn_dist.cend(), nn_dist[0]);
  // double avg_nn_dist = nn_dist[0];
  // for (size_t i = 1; i < nn_dist.size(); i++) {
  //   avg_nn_dist += nn_dist[i];
  // }
  return totalnndist / (double)nn_dist.size();
}

void standardise(std::vector<Point>& points) {
  kdt::KDTree<Point> kdtree(points);
  double minx = points[kdtree.axisFindMin(0)][0];
  double miny = points[kdtree.axisFindMin(1)][1];
  std::for_each(points.begin(), points.end(), [&](Point& p) {
    p[0] -= minx;
    p[1] -= miny;
  });
}

std::vector<Point> extended_grid(const std::vector<Point>& base,
                                 const std::vector<int>& x_edge,
                                 const std::vector<int>& y_edge,
                                 const std::vector<int>& corner, double Lx,
                                 double Ly) {
  std::vector<Point> final_grid(base.size() + x_edge.size() + y_edge.size() +
                                corner.size());
  std::copy(base.cbegin(), base.cend(), final_grid.begin());
  size_t offset = base.size();
  for (size_t i = 0; i < x_edge.size(); i++) {
    Point p = base[x_edge[i]];
    p[1] += Ly;
    final_grid[offset + i] = p;
  }
  offset += x_edge.size();
  for (size_t i = 0; i < y_edge.size(); i++) {
    Point p = base[y_edge[i]];
    p[0] += Lx;
    final_grid[offset + i] = p;
  }
  offset += y_edge.size();
  for (size_t i = 0; i < corner.size(); i++) {
    Point p = base[corner[i]];
    p[0] += Lx;
    p[1] += Ly;
    final_grid[offset + i] = p;
  }
  return final_grid;
}

void pointsToPeriodicCouplings(std::vector<Point>& points, f64 rsq,
                               std::optional<double> lx,
                               std::optional<double> ly) {
  /* This function is meant to create couplings for approximants of
   * quasicrystals. The unit cell vectors are taken to be parallel to the x and
   * y axes. If lx and/or ly have values, those are taken to be the x/y lengths
   * of the unit cell, otherwise the lengths are estimated to be the x/y extents
   * of the approximant plus the average x/y separation between neighbouring
   * points.
   */
  standardise(points);
  kdt::KDTree<Point> kdtree(points);
  double maxx = points[kdtree.axisFindMax(0)][0];
  double maxy = points[kdtree.axisFindMax(1)][1];
  kdtree.build(points);
  double avg_nn_dist = 1.0;
  if (!lx.has_value() || !ly.has_value()) {
    if (points.size() > 1) {
      avg_nn_dist = avgNNDist(kdtree, points);
    }
  }
  double Lx = lx.value_or(maxx + avg_nn_dist);
  double Ly = ly.value_or(maxy + avg_nn_dist);
  std::cout << std::format("Lx is {}\n"
                           "Ly is {}\n"
                           "Average distance to next neighbour is: {}\n",
                           Lx, Ly, avg_nn_dist);
  double search_radius = std::sqrt(rsq);
  auto x_edge =
      kdtree.axisSearch(0, (1 + 1e-8) * (search_radius - avg_nn_dist));
  auto y_edge =
      kdtree.axisSearch(1, (1 + 1e-8) * (search_radius - avg_nn_dist));
  std::vector<int> xy_corner;
  // Estimate of how many points there are in the intersection of the edges.
  xy_corner.reserve((size_t)((double)(x_edge.size() * y_edge.size()) /
                             (double)points.size()));
  for (const auto xidx : x_edge) {
    for (const auto yidx : y_edge) {
      if (xidx == yidx) {
        xy_corner.push_back(xidx);
      }
    }
  }
  auto final_grid = extended_grid(points, x_edge, y_edge, xy_corner, Lx, Ly);
  kdtree.build(final_grid);
  std::vector<Neighbour> nb_info;
  for (size_t i = 0; i < points.size(); i++) {
    auto q = final_grid[i];
    auto nbs = kdtree.radiusSearch(q, search_radius);
    for (const auto idx : nbs) {
      if ((size_t)idx > i) {
        auto p = final_grid[idx];
        Vector2d d = {p[0] - q[0], p[1] - q[1]};
        nb_info.emplace_back(i, p.idx, d);
      }
    }
  }
  MatrixXcd hamiltonian = MatrixXcd::Zero(points.size(), points.size());
  constexpr u32 ksamples = 20;
  Vector2d dual1{2 * M_PI / Lx, 0};
  Vector2d dual2{0, 2 * M_PI / Ly};
  std::vector<double> energies(ksamples * ksamples * points.size());
  auto energy_view =
      std::mdspan(energies.data(), ksamples, ksamples, points.size());
  for (u32 j = 0; j < ksamples; j++) {
    double yfrac = (double)j / ksamples;
    for (u32 i = 0; i < ksamples; i++) {
      double xfrac = (double)i / ksamples;
      update_hamiltonian(
          hamiltonian, nb_info, xfrac * dual1 + yfrac * dual2,
          [](Vector2d) { return c64{-1, 0}; }, i | j);
      SelfAdjointEigenSolver<MatrixXcd> es;
      es.compute(hamiltonian);
      for (size_t k = 0; k < points.size(); k++) {
        energy_view[i, j, k] = es.eigenvalues()[k];
        // std::cout << energies(i, j, k) << '\n';
      }
    }
  }
  hsize_t dims[3] = {ksamples, ksamples, points.size()};
  H5::DataSpace space(3, dims);
  H5::H5File file("energies.h5", H5F_ACC_TRUNC);
  H5::DataSet dataset(
      file.createDataSet("aaa", H5::PredType::NATIVE_DOUBLE, space));
  dataset.write(energies.data(), H5::PredType::NATIVE_DOUBLE);
}

MatrixXcd pointsToFiniteHamiltonian(const std::vector<Point>& points,
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

// array with space for at most N elements, but may have fewer
template <class T, size_t N>
struct MaxHeadroom {
  T data[N];
  size_t n;
  void push(T x) {
#ifndef NDEBUG
    assert(n < N);
#endif
    data[n] = x;
    n += 1;
  }
  void pop() {
#ifndef NDEBUG
    assert(n > 0);
#endif
    n -= 1;
  }
  T operator[](size_t i) const { return data[i]; }
  T& operator[](size_t i) { return data[i]; }
  T* begin() { return data; }

  T* end() { return data + n; }
  constexpr T* cbegin() noexcept { return data; }

  constexpr T* cend() noexcept { return data + n; }
};

double smallestNonZeroGap(const VectorXd& vals) {
  double min_gap = std::numeric_limits<double>::max();
  for (int i = 1; i < vals.size(); i++) {
    if (double gap = vals[i] - vals[i - 1]; gap > 1e-14 && gap < min_gap) {
      min_gap = gap;
    }
  }
  return min_gap;
}

template <class T>
struct RangeConf {
  T start;
  T end;
  u64 n;

  constexpr T d() const { return (end - start) / n; }
  constexpr T ith(uint i) const { return start + i * d(); }
};

typedef std::vector<std::vector<std::pair<double, u32>>> Delta;

Delta delta(const VectorXd& D, RangeConf<double> ec) {
  Delta delta(ec.n);
  for (u32 i = 0; i < ec.n; i++) {
    const double e = ec.ith(i);
    delta[i] = [&]() {
      std::vector<std::pair<double, u32>> tmp;
      tmp.reserve(5);
      for (u32 k = 0; k < D.size(); k++) {
        if (double diff = std::abs(D(k) - e); diff < nonzero_range) {
          tmp.push_back({std::exp(-gauss_sharpening * square(diff)), k});
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

void autoLimits(const VectorXd& D, RangeConf<double>& rc) {
  double max = D.maxCoeff();
  double min = D.minCoeff();
  double l = max - min;
  rc.start = min - 0.01 * l;
  rc.end = max + 0.01 * l;
}

std::vector<double> disp(const VectorXd& D, const MatrixXcd& UH,
                         const std::vector<Point>& points, double lat_const,
                         RangeConf<Vector2d> kc, RangeConf<double>& ec,
                         bool printProgress = true) {
  const u32 its = kc.n / 10;
  std::vector<double> disp(kc.n * ec.n, 0);
  if (fleq(ec.start, ec.end, 1e-16)) {
    autoLimits(D, ec);
  }
  auto disp_view = std::mdspan(disp.data(), ec.n, kc.n);
  if (printProgress)
    std::cout << "[";

  auto del = delta(D, ec);
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

std::vector<double> DOS(const VectorXd& D, const MatrixXcd& UH,
                        const std::vector<Point>& points, double lat_const, RangeConf<double> kxc,
                        RangeConf<double> kyc, RangeConf<double>& ec,
                        bool printProgress = true) {
  const u32 its = kxc.n / 10;
  std::vector<double> densities(ec.n, 0);
  if (fleq(ec.start, ec.end, 1e-16)) {
    autoLimits(D, ec);
  }
  auto del = delta(D, ec);
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

std::vector<double> fullSDF(const VectorXd& D, const MatrixXcd& UH,
                            const std::vector<Point>& points, double lat_const,
                            RangeConf<double> kxc, RangeConf<double> kyc,
                            RangeConf<double>& ec, bool printProgress = true) {
  std::cout << "Calculating full SDF\n";
  const u32 its = kxc.n / 10;
  std::vector<double> sdf(ec.n * kyc.n * kxc.n, 0);
  auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n, ec.n);
  if (fleq(ec.start, ec.end, 1e-16)) {
    autoLimits(D, ec);
  }
  auto del = delta(D, ec);
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

struct EigenSolution {
  VectorXd D;
  MatrixXcd U;
};

EigenSolution hermitianEigenSolver(MatrixXcd H) {
  int n = H.rows();
  const char Nchar = 'V';
  const char Uchar = 'U';
  const char Achar = 'A';
  VectorXd eigReal(n);
  const int nb = 2;
  int lwork = (nb + 1) * n * n;
  double* work = new double[lwork];
  int lrwork = 24 * n;
  double* rwork = new double[lrwork];
  int liwork = 10 * n;
  int* iwork = new int[liwork];
  int info;
  double vl, vu;
  int il, iu;
  char smin[13] = "Safe minimum";
  double abstol = dlamch_(smin);
  int M;
  MatrixXcd Z(n, n);
  int ldz = n;
  int* isuppz = new int[2 * n];
  zheevr_(&Nchar, &Achar, &Uchar, &n, (double*)H.data(), &n, &vl, &vu, &il, &iu,
          &abstol, &M, eigReal.data(), (double*)Z.data(), &ldz, isuppz, work,
          &lwork, rwork, &lrwork, iwork, &liwork, &info);
  delete[] isuppz;
  delete[] iwork;
  delete[] rwork;
  delete[] work;
  return {eigReal, Z};
}

struct SdfConf {
  // sharpness of Gaussian used to approximate Delta function
  double sharpening;
  // values below this will be removed from the Delta function.
  double cutoff;
  // In units of average nearest neighbour distance.
  double searchRadius;
  // if hasValue, do a dispersion relation with the line given by this range.
  std::optional<RangeConf<Vector2d>> DispKline;
  // Set Emin=Emax to automatically set E range
  std::optional<RangeConf<double>> DispE;
  // in units of "Brillouin zone", i.e. 1 = 2pi/a where a is a lattice constant.
  // Average nearest neighbour distance is used as a proxy for a
  RangeConf<double> SDFKx;
  RangeConf<double> SDFKy;
  // Set Emin=Emax to automatically set E range
  RangeConf<double> SDFE;
  // lattice point input (could possibly take special strings to do common
  // lattices)
  std::string pointPath;
  // Output file name
  std::string H5Filename;
  // Generally what I want. Not any more expensive than computing DOS and the
  // rest can be obtained by slicing this dataset.
  bool doFullSDF;
  // Only set if you don't want to output a full sdf to disk. Will be ignored if
  // doFullSDF is set.
  bool doDOS;
};

RangeConf<Vector2d> tblToVecRange(const toml::table& tbl) {
  toml::array start = *tbl["start"].as_array();
  toml::array end = *tbl["end"].as_array();
  return {{*start[0].as_floating_point(), *start[1].as_floating_point()}, {*end[0].as_floating_point(), *end[1].as_floating_point()}, tbl["n"].value_or<u64>(0)};
}

RangeConf<double> tblToRange(toml::table& tbl) {
  return {tbl["start"].value_or<double>(0.0), tbl["end"].value_or<double>(0.0),
          tbl["n"].value_or<u64>(0)};
}

#define SET_STRUCT_FIELD(c, tbl, key, val) c.key = tbl[#key].value_or(val)

std::optional<SdfConf> tomlToSDFConf(std::string tomlPath) {
  toml::table tbl;
  try {
    tbl = toml::parse_file(tomlPath);
  } catch (const std::exception& err) {
    std::cerr << "Parsing file " << tomlPath  << " failed with exception: " << err.what() << '\n';
    return {};
  }
  SdfConf conf{};
  toml::table& preConf = *tbl["PreConf"].as_table();
  SET_STRUCT_FIELD(conf, preConf, sharpening, 200.0);
  SET_STRUCT_FIELD(conf, preConf, cutoff, 0.1);
  SET_STRUCT_FIELD(conf, preConf, searchRadius, 1.01);
  SET_STRUCT_FIELD(conf, preConf, pointPath, "");
  SET_STRUCT_FIELD(conf, preConf, H5Filename, "SDF.h5");
  SET_STRUCT_FIELD(conf, preConf, doFullSDF, true);
  SET_STRUCT_FIELD(conf, preConf, doDOS, false);
  if (tbl["DispKline"].is_value()) {
    conf.DispKline = tblToVecRange(*tbl["DispKline"].as_table());
    conf.DispE = tblToRange(*tbl["DispE"].as_table());
  }
  conf.SDFKx = tblToRange(*tbl["SDFKx"].as_table());
  conf.SDFKy = tblToRange(*tbl["SDFKy"].as_table());
  conf.SDFE = tblToRange(*tbl["SDFE"].as_table());
  return conf;
}
#undef SET_STRUCT_FIELD

EigenSolution pointsToDiagFormHamiltonian(const std::vector<Point>& points,
                                          const kdt::KDTree<Point>& kdtree,
                                          double rad) {
  auto H = pointsToFiniteHamiltonian(points, kdtree, rad);
  EigenSolution eig = hermitianEigenSolver(H);
  return eig;
}

template <size_t n>
void writeArray(std::string s, H5::H5File& file, void* data, hsize_t sizes[n]) {
  // std::cout << "Size of parameter pack is " << sizeof(sizes) << '\n';
  // std::array<hsize_t, sizeof(sizes)> dims{sizes};
  hid_t space = H5Screate_simple(n, sizes, nullptr);
  // hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  hid_t set = H5Dcreate2(file.getId(), s.c_str(), H5T_NATIVE_DOUBLE_g, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      // file.createDataSet(s, H5::PredType::NATIVE_DOUBLE, space);
  hid_t res = H5Dwrite(set, H5T_NATIVE_DOUBLE_g, H5S_ALL, space, H5P_DEFAULT, data);
  if (res < 0) {
    std::cout << "uhoh\n";
  } 
  // writeh5wexc(set, data, H5T_STD_B64LE_g);
}

int main(const int argc, const char* const* argv) {
  cxxopts::Options options("MyProgram", "bleh");
  options.add_options()("p,points", "File name", cxxopts::value<std::string>())(
      "c,config", "TOML configuration", cxxopts::value<std::string>())(
      "t,test", "Test whatever feature I'm working on rn.",
      cxxopts::value<std::string>());

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return 1;
  }
  if (result["c"].count()) {
    std::string fname = result["c"].as<std::string>();
    SdfConf conf;
    if (auto opt = tomlToSDFConf(fname); opt.has_value()) {
      conf = opt.value();
    } else {
      return 1;
    }
    auto points = readPoints(conf.pointPath);
    kdt::KDTree kdtree(points);
    double a = avgNNDist(kdtree, points);
    auto eigsol =
        pointsToDiagFormHamiltonian(points, kdtree, conf.searchRadius * a);
    H5::H5File file = H5Fcreate(conf.H5Filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file.getId() == H5I_INVALID_HID) {
      std::cerr << "Failed to create file " << conf.H5Filename << std::endl;
    }
    if (conf.doFullSDF) {
      auto UH = eigsol.U.adjoint();
      auto sdf =
          fullSDF(eigsol.D, UH, points, a, conf.SDFKx, conf.SDFKy, conf.SDFE);
      std::cout << "dims should be: " << conf.SDFKx.n << ", " << conf.SDFKy.n << ", " <<conf.SDFE.n << '\n';
      hsize_t sizes[3] = {conf.SDFKx.n, conf.SDFKy.n,
                 conf.SDFE.n};
      writeArray<3>("sdf", file, sdf.data(), sizes);
      double sdfBounds[6] = {conf.SDFKx.start, conf.SDFKx.end,
                                         conf.SDFKy.start, conf.SDFKy.end,
                                         conf.SDFE.start,  conf.SDFE.end};
      hsize_t boundsize[1] = {6};
      writeArray<1>("sdf_bounds", file, sdfBounds, boundsize);
    } else if (conf.doDOS) {
      auto UH = eigsol.U.adjoint();
      auto dos = DOS(eigsol.D, UH, points, a, conf.SDFKx, conf.SDFKy, conf.SDFE);
      writeArray<1>("dos", file, dos.data(), &conf.SDFE.n);
      double dosBounds[2] = {conf.SDFE.start, conf.SDFE.end};
      hsize_t boundsize[1] = {2};
      writeArray<1>("dos_bounds", file, dosBounds, boundsize); 
    }
    if (conf.DispKline.has_value()) {
      if (conf.DispE.has_value()) {
        auto kc = conf.DispKline.value();
        auto ec = conf.DispE.value();
        auto UH = eigsol.U.adjoint();
        auto bleh = conf.DispKline.value();
        auto dis = disp(eigsol.D, UH, points, a, bleh, conf.DispE.value());
        hsize_t sizes[2] = {kc.n, ec.n};
        writeArray<2>("disp", file, dis.data(), sizes);
        std::array<double, 6> dispBounds = {kc.start[0], kc.start[1], kc.end[0],
                                            kc.end[1],   ec.start,    ec.end};
        hsize_t boundsizes[1] = {6};
        writeArray<1>("dos_bounds", file, dispBounds.data(), boundsizes);
      } else {
        std::cout << "Need to supply a non-zero number of energy samples\n";
      }
    }
  }
  if (result["t"].count()) {
    auto vec = readPoints(result["t"].as<std::string>());
    // constexpr size_t N = 40;
    // std::vector<Point> vec(N * N);
    // for (u32 i = 0; i < N; i++) {
    //   for (u32 j = 0; j < N; j++) {
    //     vec[N * i + j] = Point(i, j, N * i + j);
    //   }
    // }
    // pointsToSDFOnFile(vec, 1.0);
    return 0;
  }
}
