#pragma once
#include "mathhelpers.h"
#include "kdtree.h"
#include <format>
#include "Eigen/Dense"

using Eigen::Vector2d;

struct Neighbour {
  size_t i;
  size_t j;
  Vector2d d;

  friend std::ostream& operator<<(std::ostream& os, const Neighbour& nb) {
    return os << '(' << nb.i << ", " << nb.j << ", " << nb.d << ')';
  }
};

struct Point : std::array<double, 2> {
  static constexpr int DIM = 2;
  u32 idx;

  Point() {}
  Point(double x, double y, u32 idx) : idx{idx} {
    (*this)[0] = x;
    (*this)[1] = y;
  }

  constexpr Vector2d asVec() {
    return {(*this)[0], (*this)[1]};
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

std::vector<Point> readPoints(const std::string& fname);

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

