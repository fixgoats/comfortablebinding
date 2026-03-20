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

  Point() {}
  Point(f64 x, f64 y, u32 idx) : idx{idx} {
    (*this)[0] = x;
    (*this)[1] = y;
  }

  constexpr Vector2d asVec() const { return {(*this)[0], (*this)[1]}; }
  constexpr Vector2f asfVec() const { return {(*this)[0], (*this)[1]}; }

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

std::vector<Point> readPoints(const std::string& fname);
inline std::vector<Neighbour> pointsToNbs(const std::vector<Point>& points,
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
                                 const std::vector<int>& corner, double Lx,
                                 double Ly);

void standardise(std::vector<Point>& points);

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

inline MatrixXd DenseH(const std::vector<Point>& points,
                       const kdt::KDTree<Point>& kdtree, double radius,
                       f64 (*f)(Vector2d)) {

  std::vector<Neighbour> nb_info = pointsToNbs(points, kdtree, radius);
  MatrixXd H = MatrixXd::Zero(points.size(), points.size());
  for (const auto& nb : nb_info) {
    f64 val = f(nb.d);
    H(nb.i, nb.j) = val;
    H(nb.j, nb.i) = val;
  }
  return H;
}

template <class Func>
inline SparseMatrix<c64> SparseHC(const std::vector<Point>& points,
                                  const kdt::KDTree<Point>& kdtree, f64 radius,
                                  Func f) {

  std::vector<Neighbour> nb_info = pointsToNbs(points, kdtree, radius);
  SparseMatrix<c64> H(points.size(), points.size());
  for (const auto& nb : nb_info) {
    c64 val = f(nb.d);
    H.insert(nb.i, nb.j) = val;
    H.insert(nb.j, nb.i) = std::conj(val);
  }
  H.finalize();
  return H;
}

template <class Func>
inline SparseMatrix<c64> SparseHC(const Eigen::MatrixX2d& points,
                                  const Eigen::MatrixX2i& couplings, Func f) {

  SparseMatrix<c64> H(points.rows(), points.rows());
  for (s64 i = 0; i < couplings.rows(); ++i) {
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    c64 val = f(d);
    H.insert(couplings(i, 0), couplings(i, 1)) = val;
    H.insert(couplings(i, 1), couplings(i, 0)) = std::conj(val);
  }
  H.finalize();
  return H;
}

template <class Func>
inline SparseMatrix<c64> SparseC(const Eigen::MatrixX2d& points,
                                 const Eigen::MatrixX2i& couplings, Func f) {
  SparseMatrix<c64> H(points.rows(), points.rows());
  for (s64 i = 0; i < couplings.rows(); ++i) {
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    c64 val = f(d);
    H.insert(couplings(i, 0), couplings(i, 1)) = val;
    H.insert(couplings(i, 1), couplings(i, 0)) = val;
  }
  H.finalize();
  return H;
}

template <class Func>
inline SparseMatrix<c32> SparseCf(const Eigen::MatrixX2d& points,
                                  const Eigen::MatrixX2i& couplings, Func f) {
  SparseMatrix<c32> H(points.rows(), points.rows());
  for (s64 i = 0; i < couplings.rows(); ++i) {
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    c64 val = f(d);
    H.insert(couplings(i, 0), couplings(i, 1)) = val;
    H.insert(couplings(i, 1), couplings(i, 0)) = val;
  }
  H.finalize();
  return H;
}

template <class Func>
inline SparseMatrix<f64> SparseH(const std::vector<Point>& points,
                                 const kdt::KDTree<Point>& kdtree, f64 radius,
                                 Func f) {

  std::vector<Neighbour> nb_info = pointsToNbs(points, kdtree, radius);
  SparseMatrix<double> H(points.size(), points.size());
  for (const auto& nb : nb_info) {
    f64 val = f(nb.d);
    H.insert(nb.i, nb.j) = val;
    H.insert(nb.j, nb.i) = val;
  }
  H.finalize();
  return H;
}

template <class Func>
CSRMat<c64> spsh(const Eigen::MatrixX2d& points,
                 const Eigen::MatrixX2i& couplings, Func f) {
  spdlog::debug("Function template: spsh.");
  COOMat<c64> H;
  for (s64 i = 0; i < couplings.rows(); ++i) {
    spdlog::debug("writing to row: {}, col: {}", couplings(i, 0),
                  couplings(i, 1));
    Vector2d d = points(couplings(i, 1), Eigen::indexing::all) -
                 points(couplings(i, 0), Eigen::indexing::all);
    c64 val = f(d);
    H(couplings(i, 0), couplings(i, 1)) = val;
    H(couplings(i, 1), couplings(i, 0)) = val;
  }
  H.rows = points.rows();
  H.cols = points.rows();
  return CSRMat<c64>(H);
}

template <class Func>
void updatesps(CSRMat<c64>& H, const Eigen::MatrixX2d& points, Func f) {
  spdlog::debug("Function updatesps");
  for (s64 i = 0; i < H.row_indices.size() - 1; ++i) {
    u64 row_start = H.row_indices[i];
    u64 row_end = H.row_indices[i + 1];
    for (u64 j = row_start; j < row_end; ++j) {
      Vector2d d = points(i, Eigen::indexing::all) -
                   points(H.col_indices[j], Eigen::indexing::all);

      H.data[j] = f(d);
    }
  }
}
