#include "Eigen/Dense"
#include "H5Cpp.h"
#include "kdtree.h"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "typedefs.h"
#include <cxxopts.hpp>
#include <iostream>
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

/* semsagt, gera kd tré, finna punkta innan r frá y/x-ásunum, gera nýtt safn af
 * punktum með þeim punktum hliðruðum um grindarfasta til hægri, upp og bæði.
 * Gera kd tré af *nýja* punktasafninu og gera radíus leit á því. Er það ekki
 * bara fínt? Er nokkuð rosa dýrt að gera kd tré tvisvar? En bíddu við, ætti ég
 * ekki að geta bætt "nýju" punktunum við í staðinn fyrir að gera nýtt tré?
 * Eiginlega ekki, ég þarf alltaf að búa til nýtt tré frá grunni. Gæti fundið
 * meðal NN fjarlægðirnar og lagt þær saman við lengdirnar til að meta
 * grindarfastana.
 */

/* Skema:
 * punktar í textaskrá -> vigur af punktum með indexum -> kd-tré af punktum
 * -> finna meðalfjarlægð NN + finna alla punkta innan r frá ásum (Y, X og Z)
 * -> bæta Y, X, og Z við mengið, hliðrað um grindarvigrana m. vi. hætti.
 * -> endurgera kd tréð. -> gera radíus leit til að finna næstu nágranna.
 * -> gera vigur af nágrannaupplýsingum (gæti verið vigur af (i, j, \vec{r}) eða
 *  vigur af vigrum af (j, \vec{r}). vigur af vigrum myndi taka minna pláss ef
 *  ef það eru a.m.k. 3 tengingar? allavega ef það eru margar tengingar þá fer
 *  það að borga sig en ég held að í mínu tilfelli sé vigur af (i, j, \vec{r})
 *  best.)
 * -> Nota nágrannaupplýsingar til að gera H(k) eins og er gert í pythtb.
 * -> Ólíkt PythTB get ég endurnýtt minnið sem Hamilton fylkið notar í staðinn
 *  fyrir að deallocatea og allocatea.
 * -> þráðun: margþráðaður eigingilda-algóriþmi? væri ekki skilvirkara að keyra
 *  einn algoriþma á hverjum þræði? Setja markmiðs-þráðafjölda, en ganga úr
 * skugga um að forritið noti ekki of mikið vinnsluminni og kannski ekki alveg
 * alla kjarnana á tölvunni nema notandinn biðji um það.
 */

void pointsToPeriodicCouplings(const MatrixX2d& points, f64 rsq,
                               std::optional<double> lx,
                               std::optional<double> ly) {
  /* This function is meant to create couplings for approximants of
   * quasicrystals. The unit cell vectors are taken to be parallel to the x and
   * y axes. If lx and/or ly have values, those are taken to be the x/y lengths
   * of the unit cell, otherwise the lengths are estimated to be the x/y extents
   * of the approximant plus the average x/y separation between neighbouring
   * points (actually that sounds rather complicated, I'd have to iterate over
   * the whole lattice, finding the neighbours, and then... actually that might
   * be fine, if I do one pass for the approximant and then somehow handle edge
   * cases... or! I could use rsq!) So, if I turn one unit cell into four, I'll
   * have to check which cell the neighbour is in and correlate the neighbour
   * with an "atom" in the original cell. So, I might filter out the points that
   * are less than r from the x-axis, the y-axis and their intersection and only
   * add those to the extended grid. Make a k-d tree?
   */
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

/* int main(const int argc, const char* const* argv) {
  output<f64>("../data/testfile.txt");
  output<c64>("../data/complexfile.txt");
  auto m = readEigen<f64>("../data/testfile.txt");
  std::cout << m << std::endl;
} */

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
    H(nb.j, nb.i) += val;
  }
}

template <class T>
struct Arr3D : std::vector<T> {
  size_t x;
  size_t y;
  size_t z;

  Arr3D(size_t x, size_t y, size_t z)
      : std::vector<T>(x * y * z), x{x}, y{y}, z{z} {}

  T& operator()(size_t i, size_t j, size_t k) {
    return (*this)[y * z * i + z * j + k];
  }

  T operator()(size_t i, size_t j, size_t k) const {
    return (*this)[y * z * i + z * j + k];
  }
};

constexpr size_t sum(std::convertible_to<size_t> auto... i) {
  return (0 + ... + i);
}

constexpr size_t product(std::convertible_to<size_t> auto... i) {
  return (0 * ... * i);
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

int main(const int argc, const char* const* argv) {
  ArrND<double> A(2, 3, 4);
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
    standardise(vec);
    kdt::KDTree<Point> kdtree(vec);
    double maxx = vec[kdtree.axisFindMax(0)][0];
    double maxy = vec[kdtree.axisFindMax(1)][1];
    kdtree.build(vec);
    double avg_nn_dist = 1.0;
    if (vec.size() > 1) {
      avg_nn_dist = avgNNDist(kdtree, vec);
    }
    double Lx = maxx + avg_nn_dist;
    double Ly = maxy + avg_nn_dist;
    std::cout << std::format("Lx is {}\n"
                             "Ly is {}\n"
                             "Average distance to next neighbour is: {}\n",
                             Lx, Ly, avg_nn_dist);
    double search_radius = 2.01 * avg_nn_dist;
    auto x_edge = kdtree.axisSearch(0, search_radius - avg_nn_dist + 1e-6);
    auto y_edge = kdtree.axisSearch(1, search_radius - avg_nn_dist + 1e-6);
    std::vector<int> xy_corner;
    // Estimate of how many points there are in the intersection of the edges.
    xy_corner.reserve(
        (size_t)((double)(x_edge.size() * y_edge.size()) / (double)vec.size()));
    for (const auto xidx : x_edge) {
      for (const auto yidx : y_edge) {
        if (xidx == yidx) {
          xy_corner.push_back(xidx);
        }
      }
    }
    std::vector<Point> final_grid(vec.size() + x_edge.size() + y_edge.size() +
                                  xy_corner.size());
    std::copy(vec.cbegin(), vec.cend(), final_grid.begin());
    size_t offset = vec.size();
    for (size_t i = 0; i < x_edge.size(); i++) {
      Point p = vec[x_edge[i]];
      p[1] += Ly;
      final_grid[offset + i] = p;
    }
    offset += x_edge.size();
    for (size_t i = 0; i < y_edge.size(); i++) {
      Point p = vec[y_edge[i]];
      p[0] += Lx;
      final_grid[offset + i] = p;
    }
    offset += y_edge.size();
    for (size_t i = 0; i < xy_corner.size(); i++) {
      Point p = vec[xy_corner[i]];
      p[0] += Lx;
      p[1] += Ly;
      final_grid[offset + i] = p;
    }
    kdtree.build(final_grid);
    std::vector<Neighbour> nb_info;
    for (size_t i = 0; i < vec.size(); i++) {
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
    MatrixXcd hamiltonian = MatrixXcd::Zero(vec.size(), vec.size());
    constexpr u32 ksamples = 20;
    Vector2d dual1{2 * M_PI / Lx, 0};
    Vector2d dual2{0, 2 * M_PI / Ly};
    Arr3D<double> energies(ksamples, ksamples, vec.size());
    for (u32 j = 0; j < ksamples; j++) {
      double yfrac = (double)j / ksamples;
      for (u32 i = 0; i < ksamples; i++) {
        double xfrac = (double)i / ksamples;
        update_hamiltonian(
            hamiltonian, nb_info, xfrac * dual1 + yfrac * dual2,
            [](Vector2d) { return c64{-1, 0}; }, i | j);
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(hamiltonian);
        for (size_t k = 0; k < vec.size(); k++) {
          energies(i, j, k) = es.eigenvalues()[k];
          // std::cout << energies(i, j, k) << '\n';
        }
      }
    }
    hsize_t dims[3] = {ksamples, ksamples, vec.size()};
    H5::DataSpace space(3, dims);
    H5::H5File file("energies.h5", H5F_ACC_TRUNC);
    H5::DataSet dataset(
        file.createDataSet("aaa", H5::PredType::NATIVE_DOUBLE, space));
    dataset.write(energies.data(), H5::PredType::NATIVE_DOUBLE);

    return 0;
  }
  if (result["p"].count() && result["r"].count()) {
    MatrixX2d m = readEigen<f64>(result["p"].as<std::string>());
    MatrixXd H =
        couplingmat(m, result["r"].as<f64>(), [](f64 x) { return -exp(-x); });

    Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(H);

    if (eigensolver.info() != Eigen::Success) {
      std::cout << eigensolver.info() << std::endl;
      abort();
    }
    std::ofstream f("eigvals.txt");
    f << eigensolver.eigenvalues().format(defaultFormat);
    f.close();
    f64 xmin = m(Eigen::all, 0).minCoeff();
    f64 ymin = m(Eigen::all, 1).minCoeff();
    f64 xmax = m(Eigen::all, 0).maxCoeff();
    f64 ymax = m(Eigen::all, 1).maxCoeff();
    f64 Lx = xmax - xmin;
    f64 Ly = ymax - ymin;
  }
  if (result["s"].count()) {
    MatrixXd m = readEigen<f64>(result["p"].as<std::string>());
    MatrixXd H =
        couplingmat(m, result["r"].as<f64>(), [](f64) { return f64{-1.0}; });

    Eigen::VectorXcd psi;
    if (result["i"].count()) {
      psi = readEigen<c64>(result["i"].as<std::string>());
    } else {
      std::random_device dev;
      std::mt19937_64 mt(dev());
      std::uniform_real_distribution<f64> dis(0, M_2_PI);
      for (u32 i = 0; i < H.rows(); i++) {
        psi(i) = c64{cos(dis(mt)), sin(dis(mt))};
      }
    }
  }
}
