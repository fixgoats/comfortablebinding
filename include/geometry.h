#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "kdtree.h"
#include "mathhelpers.h"
#include "spdlog/spdlog.h"
#include <format>

using Eigen::Vector2d, Eigen::Vector2f, Eigen::MatrixXd, Eigen::SparseMatrix,
    Eigen::Vector2i, Eigen::MatrixXcd, Eigen::VectorXcd;

struct Neighbour {
  size_t i;
  size_t j;
  Vector2d d;

  friend std::ostream& operator<<(std::ostream& os, const Neighbour& nb) {
    return os << '(' << nb.i << ", " << nb.j << ", " << nb.d << ')';
  }
};

struct Point : std::array<f64, 2> {
  static constexpr s64 DIM = 2;
  u32 idx;

  Point() = default;
  Point(f64 x, f64 y, u32 idx) : idx{idx} {
    (*this)[0] = x;
    (*this)[1] = y;
  }

  [[nodiscard]] constexpr Vector2d as_vec() const {
    return {(*this)[0], (*this)[1]};
  }
  [[nodiscard]] constexpr Vector2f as_fvec() const {
    return {(*this)[0], (*this)[1]};
  }

  [[nodiscard]] double sqdist(const Point& p) const {
    return square((*this)[0] - p[0]) + square((*this)[1] - p[1]);
  }

  [[nodiscard]] double dist(const Point& p) const {
    return sqrt(square((*this)[0] - p[0]) + square((*this)[1] - p[1]));
  }

  friend std::ostream& operator<<(std::ostream& os, const Point& pt) {
    return os << std::format("({}, {}, {})", pt[0], pt[1], pt.idx);
  }
};

struct Pt2 : std::array<f64, 2> {
  static const s64 DIM = 2;
  [[nodiscard]] f64 sqdist(const Pt2& p) const {
    return square((*this)[0] - p[0]) + square((*this)[1] - p[1]);
  }

  [[nodiscard]] double dist(const Pt2& p) const {
    return sqrt(square((*this)[0] - p[0]) + square((*this)[1] - p[1]));
  }
};

std::vector<Point> read_points(const std::string& fname);
inline std::vector<Neighbour> points_to_nbs(const std::vector<Point>& points,
                                            const kdt::KDTree<Point>& kdtree,
                                            f64 radius) {
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
  return nb_info;
}

std::vector<Point> extended_grid(const std::vector<Point>& base,
                                 const std::vector<int>& x_edge,
                                 const std::vector<int>& y_edge,
                                 const std::vector<int>& corner, double lx,
                                 double ly);

void standardise(std::vector<Point>& points);

template <class PointT>
double max_norm_dist(const PointT& q, const PointT& p) {
  double max = 0;
  for (size_t i = 0; i < PointT::DIM; i++) {
    double d = abs(q[i] - p[i]);
    max = std::max(max, d);
  }
  return max;
}

template <class PointT>
double avg_nn_dist(kdt::KDTree<PointT>& kdtree,
                   const std::vector<PointT>& points) {
  std::vector<double> nn_dist(points.size());
  for (size_t i = 0; i < points.size(); i++) {
    int idx = kdtree.knnSearch(points[i], 2)[1];
    nn_dist[i] = points[i].dist(points[idx]);
  }
  double totalnndist =
      std::accumulate(nn_dist.cbegin() + 1, nn_dist.cend(), nn_dist[0]);
  return totalnndist / (double)nn_dist.size();
}

template <class PointT>
f64 max_nn_dist(kdt::KDTree<PointT>& kdtree,
                const std::vector<PointT>& points) {
  std::vector<f64> nn_dist(points.size());
  for (size_t i = 0; i < points.size(); i++) {
    int idx = kdtree.knnSearch(points[i], 2)[1];
    nn_dist[i] = points[i].dist(points[idx]);
  }
  f64 totalnndist = std::accumulate(nn_dist.cbegin() + 1, nn_dist.cend(), 0,
                                    std::greater_equal<>());
  return totalnndist / (double)nn_dist.size();
}

inline MatrixXd dense_h(const std::vector<Point>& points,
                        const kdt::KDTree<Point>& kdtree, double radius,
                        f64 (*f)(Vector2d)) {

  std::vector<Neighbour> nb_info = points_to_nbs(points, kdtree, radius);
  MatrixXd ham = MatrixXd::Zero(static_cast<s64>(points.size()),
                                static_cast<s64>(points.size()));
  for (const auto& nb : nb_info) {
    f64 val = f(nb.d);
    ham(static_cast<s64>(nb.i), static_cast<s64>(nb.j)) = val;
    ham(static_cast<s64>(nb.j), static_cast<s64>(nb.i)) = val;
  }
  return ham;
}

template <class Func>
inline SparseMatrix<c64> sparse_hc(const std::vector<Point>& points,
                                   const kdt::KDTree<Point>& kdtree, f64 radius,
                                   Func f) {

  std::vector<Neighbour> nb_info = points_to_nbs(points, kdtree, radius);
  SparseMatrix<c64> ham(static_cast<s64>(points.size()),
                        static_cast<s64>(points.size()));
  for (const auto& nb : nb_info) {
    c64 val = f(nb.d);
    ham.insert(static_cast<s64>(nb.i), static_cast<s64>(nb.j)) = val;
    ham.insert(static_cast<s64>(nb.j), static_cast<s64>(nb.i)) = std::conj(val);
  }
  ham.finalize();
  return ham;
}

template <class Func>
inline SparseMatrix<c64> sparse_hc(const Eigen::MatrixX2d& points,
                                   const Eigen::MatrixX2i& couplings, Func f) {

  SparseMatrix<c64> ham(points.rows(), points.rows());
  for (s64 i = 0; i < couplings.rows(); ++i) {
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    c64 val = f(d);
    ham.insert(couplings(i, 0), couplings(i, 1)) = val;
    ham.insert(couplings(i, 1), couplings(i, 0)) = std::conj(val);
  }
  ham.finalize();
  return ham;
}

template <class Func>
inline SparseMatrix<c64> sparse_c(const Eigen::MatrixX2d& points,
                                  const Eigen::MatrixX2i& couplings, Func f) {
  SparseMatrix<c64> ham(points.rows(), points.rows());
  for (s64 i = 0; i < couplings.rows(); ++i) {
    spdlog::debug("Pair: {}, {}", couplings(i, 0), couplings(i, 1));
    spdlog::debug("subtracting ({}, {}) from ({}, {})",
                  points(couplings(i, 0), 0), points(couplings(i, 0), 1),
                  points(couplings(i, 1), 0), points(couplings(i, 1), 1));
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    spdlog::debug("Distance is: {}", d.norm());
    c64 val = f(d);
    ham.insert(couplings(i, 0), couplings(i, 1)) = val;
    ham.insert(couplings(i, 1), couplings(i, 0)) = val;
  }
  ham.finalize();
  return ham;
}

template <class Func>
inline SparseMatrix<c32> sparse_cf(const Eigen::MatrixX2d& points,
                                   const Eigen::MatrixX2i& couplings, Func f) {
  SparseMatrix<c32> ham(points.rows(), points.rows());
  for (s64 i = 0; i < couplings.rows(); ++i) {
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    c64 val = f(d);
    ham.insert(couplings(i, 0), couplings(i, 1)) = val;
    ham.insert(couplings(i, 1), couplings(i, 0)) = val;
  }
  ham.finalize();
  return ham;
}

template <class Func>
inline SparseMatrix<f64> sparse_h(const std::vector<Point>& points,
                                  const kdt::KDTree<Point>& kdtree, f64 radius,
                                  Func f) {

  std::vector<Neighbour> nb_info = points_to_nbs(points, kdtree, radius);
  SparseMatrix<double> ham(static_cast<s64>(points.size()),
                           static_cast<s64>(points.size()));
  for (const auto& nb : nb_info) {
    f64 val = f(nb.d);
    ham.insert(static_cast<s64>(nb.i), static_cast<s64>(nb.j)) = val;
    ham.insert(static_cast<s64>(nb.j), static_cast<s64>(nb.i)) = val;
  }
  ham.finalize();
  return ham;
}

template <class Func>
CSRMat<c64> spsh(const Eigen::MatrixX2d& points,
                 const Eigen::MatrixX2i& couplings, Func f) {
  spdlog::debug("Function template: spsh.");
  COOMat<c64> ham;
  for (s64 i = 0; i < couplings.rows(); ++i) {
    spdlog::debug("writing to row: {}, col: {}", couplings(i, 0),
                  couplings(i, 1));
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    c64 val = f(d);
    ham(couplings(i, 0), couplings(i, 1)) = val;
    ham(couplings(i, 1), couplings(i, 0)) = val;
  }
  ham.rows = points.rows();
  ham.cols = points.rows();
  return CSRMat<c64>(ham);
}

template <class Func>
void updatesps(CSRMat<c64>& ham, const Eigen::MatrixX2d& points, Func f) {
  spdlog::debug("Function updatesps");
  for (s64 i = 0; i < ham.row_indices.size() - 1; ++i) {
    u64 row_start = ham.row_indices[i];
    u64 row_end = ham.row_indices[i + 1];
    for (u64 j = row_start; j < row_end; ++j) {
      Vector2d d = points(i, Eigen::indexing::all) -
                   points(ham.col_indices[j], Eigen::indexing::all);

      ham.data[j] = f(d);
    }
  }
}
