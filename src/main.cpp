#include "Eigen/Dense"
#include "H5Cpp.h"
#include "kdtree.h"
#include "lodepng.h"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "typedefs.h"
#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <cxxopts.hpp>
#include <iostream>
#include <mdspan>
#include <random>
#include <ranges>
#include <sstream>

using Eigen::MatrixXd, Eigen::MatrixXcd, Eigen::VectorXd, Eigen::MatrixX2d,
    Eigen::VectorXcd, Eigen::Vector2d, Eigen::SelfAdjointEigenSolver;
constexpr u32 Nx = 40;
constexpr u32 Ny = 40;
static Eigen::IOFormat defaultFormat(Eigen::StreamPrecision,
                                     Eigen::DontAlignCols, " ", "\n", "", "",
                                     "", "");

template <class Func>
MatrixXcd couplingmat(const VectorXd& xs, const VectorXd& ys, Vector2d k,
                      f64 rsq0, Func f) {
  const u32 n = xs.size();
  std::cout << n << '\n';
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

/// nx: number of hexagons along x axis.
/// ny: number of hexagons along y axis.
/// a: lattice constant.
/*
 * void makeHexLattice(u32 nx, u32 ny, f64 a) {
  Eigen::Vector2d v1{a * std::cos(M_2_PI / 3), a * std::cos(M_2_PI / 3)};
  Eigen::Vector2d v2{a * std::cos(M_PI / 3), a * std::cos(M_PI / 3)};
  Eigen::Vector2d h{a, 0};
  u32 npoints = 2 * (2 * ny + 1) + (nx - 1) * (2 * ny + 2);

  // Eigen::MatrixX2d = ;
  Eigen::MatrixX2d H(npoints, 2);
  if (nx % 2 == 1) {
    u32 idx = 0;
    for (u32 i = 0; i < 2 * ny + 1; i++) {
      H(idx, {1, 2}) = (i + 1 / 2) * v1 + (i / 2) * v2;
      idx += 1;
    }

    for (u32 j = 1; j < nx; j++) {
      for (u32 i = 0; i < 2 * ny + 1; i++) {
        H(idx, {1, 2}) = (i + 1 / 2) * v1 + (i / 2) * v2;
        idx += 1;
      }
    }
  }
  H(0, {1, 2}) = 0 * v1 + 0 * v2;
  H(1, {1, 2}) = 1 * v1 + 0 * v2;
  H(2, {1, 2}) = 1 * v1 + 1 * v2;
  if (nx % 2 == 0) {
    for (u32 i = 0; i < nx + 1; i++) {
      H(2 * i * (2 * ny + 1), {1, 2}) = 2 * i * h + (i / 2) * v2;
      for (u32 j = 0; j < ny; j++) {
        H(2 * i * (2 * ny + 1) + 2 * j, {1, 2}) = (i / 2) * v2
      }
    }
    for (u32 i = 0; i < nx; i++) {
      H(i * (2 * ny + 1), {1, 2});
      for (u32 j = 0; j < ny; j++) {
      }
    }
  }
}
*/

template <class Func>
MatrixXd couplingmat(const MatrixX2d& points, f64 rsq, Func f) {
  const u32 n = points.rows();
  std::cout << n << '\n';
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

MatrixXcd pointsToFiniteHamiltonian(std::vector<Point>& points, f64 rsq) {
  /* This function creates a hamiltonian for a simple finite lattice.
   * Can't exactly do a dispersion from this.
   */
  standardise(points);
  kdt::KDTree<Point> kdtree(points);
  kdtree.build(points);
  double search_radius = std::sqrt(rsq);
  std::vector<Neighbour> nb_info;
  for (size_t i = 0; i < points.size(); i++) {
    auto q = points[i];
    auto nbs = kdtree.radiusSearch(q, search_radius);
    for (const auto idx : nbs) {
      if ((size_t)idx > i) {
        auto p = points[idx];
        Vector2d d = {p[0] - q[0], p[1] - q[1]};
        nb_info.emplace_back(i, p.idx, d);
      }
    }
  }
  return finite_hamiltonian(points.size(), nb_info,
                            [](Vector2d) { return -1; });
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

struct Delta {
  // indices corresponding to where the delta is non-zeros
  std::vector<size_t> indices;
  // offset[i] corresponds to the end of the range in indices corresponding to
  // energy value no. i
  std::vector<size_t> offsets;

  Delta(size_t n) : indices(n), offsets(n) {}
};

// Assumes vals are ordered, exits once the search can't yield more indices.
// Accepts a begin parameter so you don't always have to start searching from
// the beginning
Delta delta(const VectorXd& vals, double tol, double ebegin, double eend,
            size_t n) {
  Delta indices(n);
  double de = (eend - ebegin) / n;
  size_t val_start = 0;
  for (size_t i = 0; i < n; i++) {
    double e = ebegin + i * de;
    if (i > 0)
      indices.offsets[i] = indices.offsets[i - 1];
    for (size_t j = val_start; j < (size_t)vals.size(); j++) {
      if (std::abs(vals[j] - e) < tol) {
        indices.indices.push_back(j);
        indices.offsets[i] += 1;
      } else if (vals[j] > e) {
        break;
      } else
        val_start = j;
    }
  }
  return indices;
}

MatrixXd delta(const VectorXd& eigvals, double e, const MatrixXcd& U,
               const MatrixXcd& UH) {
  const double range = eigvals(Eigen::indexing::last) - eigvals(0);
  const double error = range / (2 * (double)eigvals.size());
  const size_t n = [&]() {
    size_t i = 0;
    for (const auto& x : eigvals) {
      if (std::abs(x - e) < error)
        break;
      i += 1;
    }
    return i;
  }();
  // some member of eigvals was close enough to e
  // Might need refinement for degenerate eigvals, but this should be super fast
  if (n < (size_t)eigvals.size()) {
    return (U(n, Eigen::indexing::all) * UH(Eigen::indexing::all, n)).real();
  } else {
    return MatrixXd::Zero(eigvals.size(), eigvals.size());
  }
}

double smallestNonZeroGap(const VectorXd& vals) {
  double min_gap = std::numeric_limits<double>::max();
  for (int i = 1; i < vals.size(); i++) {
    if (double gap = vals[i] - vals[i - 1]; gap > 1e-14 && gap < min_gap) {
      min_gap = gap;
    }
  }
  return min_gap;
}

// Not sure whether I should insert rsq here or find it outside the function.
void spectralDensityFunction(std::vector<Point>& points, f64) {
  kdt::KDTree<Point> kdtree(points);
  std::cout << "Finding average nearest neighbour distance\n";
  const double a = avgNNDist(kdtree, points);
  std::cout << "Making Hamiltonian\n";
  const auto hamiltonian = pointsToFiniteHamiltonian(points, 4 * a * a);
  std::cout << "Diagonalizing Hamiltonian\n";
  Eigen::BDCSVD<MatrixXcd> svd(hamiltonian);
  // Eigen::SelfAdjointEigenSolver<MatrixXcd> eigensolver(hamiltonian);
  // const MatrixXcd U = eigensolver.eigenvectors();
  const MatrixXcd U = svd.matrixU();
  // const VectorXd D = eigensolver.eigenvalues();
  const VectorXd D = svd.singularValues();
  const size_t ne = 100;
  std::cout << "Finding smallest gap\n";
  const double tol = smallestNonZeroGap(D);
  const Delta del = delta(D, tol, D(0), D(Eigen::indexing::last), ne);
  // MatrixXcd UH = U.adjoint();
  const size_t nk = 100;
  const u32 its = nk / 10;
  const double kmax = M_PI / a;
  const double dkx = 2 * kmax / nk;
  const double emin = -4.;
  const double emax = 4.;
  const double de = (emax - emin) / ne;
  std::vector<double> densities(nk * ne, 0);
  auto density_view = std::mdspan(densities.data(), ne, nk);
  std::cout << "Calculating SDF\n";
  std::cout << '[';
  for (size_t i = 0; i < nk; i++) {
    const double kx = -kmax + i * dkx;
    const VectorXcd k_vec = [&]() {
      VectorXcd tmp = VectorXcd::Zero(points.size());
      std::transform(points.begin(), points.end(), tmp.begin(),
                     [&](Point p) { return std::exp(c64{0, -kx * p[0]}); });
      tmp = U * k_vec;
      return tmp;
    }();
    for (size_t j = 0; j < ne; j++) {
      const size_t begin = j > 0 ? del.offsets[j - 1] : 0;
      const size_t end = del.offsets[j];
      for (size_t k = begin; k < end; k++) {
        density_view[j, i] += std::norm(k_vec(del.indices[k]));
      }
    }
    if (i % its == 0) {
      std::cout << "█|";
    }
  }
  std::cout << "█]";
  // Er þetta komið? held það
  hsize_t dims[2] = {ne, nk};
  H5::DataSpace space(2, dims);
  H5::H5File file("densities.h5", H5F_ACC_TRUNC);
  H5::DataSet dataset(
      file.createDataSet("aaa", H5::PredType::NATIVE_DOUBLE, space));
  dataset.write(densities.data(), H5::PredType::NATIVE_DOUBLE);
}

// dos: finna minnsta og lægsta eigingildi, og scaling þætti a, b.
// a = (E_max - E_min) / (2 - epsilon)
// b = (E_max + E_min) / 2
/*
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
*/

int main(const int argc, const char* const* argv) {
  cxxopts::Options options("MyProgram", "bleh");
  options.add_options()("p,points", "File name", cxxopts::value<std::string>())(
      "c,chain", "Make chain points", cxxopts::value<u32>())(
      "r,radius", "Maximum separation between points to consider",
      cxxopts::value<f64>())("s,sim", "Do dynamic simulation")(
      "t,test", "Test whatever feature I'm working on rn.",
      cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);
  if (result["c"].count()) {
    u32 n = result["c"].as<u32>();
    MatrixX2d points(n, 2);
    for (u32 i = 0; i < n; i++) {
      points(i, {0, 1}) = Eigen::Vector2d{i, 0};
    }
    saveEigen("chain.txt", points);
  }
  if (result["t"].count()) {
    auto vec = readPoints(result["t"].as<std::string>());
    /*std::vector<Point> vec(20 *
                           20); //
    for (u32 i = 0; i < 20; i++) {
      for (u32 j = 0; j < 20; j++) {
        vec[20 * i + j] = Point(i, j, 20 * i + j);
      }
    }*/
    spectralDensityFunction(vec, 1.0);
    return 0;
  }
  if (result["p"].count()) {
    hsize_t size = {};
  }
}
