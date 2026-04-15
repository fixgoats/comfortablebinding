#include "Eigen/Core"
#include "Eigen/Dense"
#include "highfive/highfive.hpp"
#include "spdlog/spdlog.h"
#include "typedefs.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cxxopts.hpp>
#include <execution>
#include <iostream>
#include <random>
#include <variant>
#include <vector>

using Eigen::Vector3d, Eigen::Vector2d, Eigen::Matrix3Xd, Eigen::Matrix3d,
    Eigen::Matrix2d;

template <class T>
struct Node {
  std::vector<std::shared_ptr<Node<T>>> children;
  T data{};

  Node(T obj) : data{obj} { std::cout << "Node(T) constructor.\n"; }
  Node(std::vector<std::shared_ptr<Node<T>>> ch, T obj)
      : children{ch}, data{obj} {}
};

template <class T>
struct Tree {
  std::shared_ptr<Node<T>> root;
  Tree<T>(Node<T>* r) : root{r} {
    std::cout << "Tree<T>(Node<T>) constructor.\n";
  }
  Tree<T>(std::shared_ptr<Node<T>> r) : root{r} {}
  Tree<T>(T r) : root{std::make_shared<Node<T>>(r)} {}
};

Tree<u64> add_trees(Tree<u64>& a, Tree<u64>& b) {
  spdlog::debug("Function: add_trees.");
  spdlog::debug("Making array of pointers to a and b roots.");
  return Tree<u64>{
      new Node<u64>{{a.root, b.root}, a.root->data + b.root->data}};
}

template <class... Ts>
struct overloads : Ts... {
  using Ts::operator()...;
};

constexpr double sq3 = 1.7320508075688772935;

static const Eigen::Matrix<f64, 3, 13> hat =
    (Eigen::Matrix<f64, 3, 13>() << 0., -0.75, -0.5, 0.5, 0.75, 1.5, 2.25, 2.,
     1.5, 1.5, 0.75, 0.5, 0., 0., -0.25 * sq3, -0.5 * sq3, -0.5 * sq3,
     -0.25 * sq3, -0.5 * sq3, -0.25 * sq3, 0., 0., 0.5 * sq3, 0.75 * sq3,
     0.5 * sq3, 0.5 * sq3, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.)
        .finished();
// c64{0., 0.}, c64{-0.75, -0.25 * sq3}, c64{-0.5, -0.5 * sq3},
//     c64{0.5, -0.5 * sq3}, c64{0.75, -0.25 * sq3}, c64{1.5, -0.5 * sq3},
//     c64{2.25, -0.25 * sq3}, c64{2., 0.}, c64{1.5, 0.}, c64{1.5, 0.5 * sq3},
//     c64{0.75, 0.75 * sq3}, c64{0.5, 0.5 * sq3}, c64 {
//   0., 0.5 * sq3
// }

static const Eigen::Matrix<f64, 3, 13> hat1 =
    (Eigen::Matrix<f64, 3, 13>() << 0., 0.75, 0.5, -0.5, -0.75, -1.5, -2.25,
     -2., -1.5, -1.5, -0.75, -0.5, 0., 0., -0.25 * sq3, -0.5 * sq3, -0.5 * sq3,
     -0.25 * sq3, -0.5 * sq3, -0.25 * sq3, 0., 0., 0.5 * sq3, 0.75 * sq3,
     0.5 * sq3, 0.5 * sq3, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.)
        .finished();

constexpr Matrix3d aff_rot(f64 ang) {
  return (Matrix3d() << cos(ang), -sin(ang), 0., sin(ang), cos(ang), 0., 0., 0.,
          1.)
      .finished();
}

constexpr Matrix3d sixth_rot(u64 i) {
  switch (i) {
  case 0:
    return Matrix3d::Identity();
  case 1:
    return (Matrix3d() << 0.5, -sq3 / 2., 0., sq3 / 2., 0.5, 0., 0., 0., 1.)
        .finished();
  case 2:
    return (Matrix3d() << -0.5, -sq3 / 2., 0., sq3 / 2., -0.5, 0., 0., 0., 1.)
        .finished();
  case 3:
    return (Matrix3d() << -1., 0., 0., 0., -1., 0., 0., 0., 1.).finished();
  case 4:
    return (Matrix3d() << -0.5, sq3 / 2., 0., -sq3 / 2., -0.5, 0., 0., 0., 1.)
        .finished();
  case 5:
    return (Matrix3d() << 0.5, sq3 / 2., 0., -sq3 / 2., 0.5, 0., 0., 0., 1.)
        .finished();
  default:
    return sixth_rot(i % 6);
  }
}

constexpr Matrix3d transl2(Vector2d v) {
  return (Matrix3d() << 1., 0., v[0], 0., 1., v[1], 0., 0., 1.).finished();
}

constexpr Matrix3d transl3(Vector3d v) {
  return (Matrix3d() << 1., 0., v[0], 0., 1., v[1], 0., 0., 1.).finished();
}

constexpr Matrix3d rot_about(Vector3d v, u64 ang) {
  return transl3(v) * (sixth_rot(ang) * transl3(-v));
}

constexpr Matrix3d from_seg(Vector3d p, Vector3d q) {
  return (Matrix3d() << q[0] - p[0], p[1] - q[1], p[0], q[1] - p[1],
          q[0] - p[0], p[1], 0., 0., 1.)
      .finished();
}

constexpr Matrix3d aff_inv(Matrix3d aff) {
  Matrix2d mat_part =
      (Matrix2d() << aff(0, 0), aff(0, 1), aff(1, 0), aff(1, 1)).finished();
  Matrix2d mat_part_inv = mat_part.inverse();
  Vector2d transl_part = -mat_part_inv * Vector2d{aff(0, 2), aff(1, 2)};
  return (Matrix3d() << mat_part_inv(0, 0), mat_part_inv(0, 1), transl_part[0],
          mat_part_inv(1, 0), mat_part_inv(1, 1), transl_part[1], 0., 0., 1.)
      .finished();
}

constexpr Matrix3d match_segs(Vector3d p1, Vector3d q1, Vector3d p2,
                              Vector3d q2) {
  return from_seg(p2, q2) * (aff_inv(from_seg(p1, q1)));
}

constexpr Matrix3d translate_by(Matrix3d aff, Vector2d v) {
  aff(0, 2) += v[0];
  aff(1, 2) += v[1];
  return aff;
}

Vector3d intersection(Vector3d p1, Vector3d q1, Vector3d p2, Vector3d q2) {
  const f64 d =
      (q2[1] - p2[1]) * (q1[0] - p1[0]) - (q2[0] - p2[0]) * (q1[1] - p1[1]);
  const f64 u_a = ((q2.x() - p2.x()) * (p1.y() - p2.y()) -
                   (q2.y() - p2.y()) * (p1.x() - p2.x())) /
                  d;
  return (Vector3d() << p1.x() + u_a * (q1.x() - p1.x()),
          p1.y() + u_a * (q1.y() - p1.y()), 1.)
      .finished();
}

constexpr Vector3d affsub(Vector3d v, Vector3d u) {
  return {v[0] - u[0], v[1] - u[1], 1.};
}

constexpr Vector3d affadd(Vector3d v, Vector3d u) {
  return {v[0] + u[0], v[1] + u[1], 1.};
}
struct Base {};

typedef std::variant<Eigen::Matrix<f64, 3, 3>, Eigen::Matrix<f64, 3, 4>,
                     Eigen::Matrix<f64, 3, 5>, Eigen::Matrix<f64, 3, 6>>
    shape_var_t;
struct ShapeMaker {
  Matrix3d transform;
  // union Shape {
  // Eigen::Matrix<f64, 3, 3> tri;
  // Eigen::Matrix<f64, 3, 4> para;
  // Eigen::Matrix<f64, 3, 5> penta;
  // Eigen::Matrix<f64, 3, 6> hexa;
  // Eigen::Matrix<f64, 3, 13> hat;
  // Eigen::Matrix<f64, 3, 13> hat1;
  // };
  // std::shared_ptr<Shape> p_shape;

  std::shared_ptr<shape_var_t> p_shape;

  Matrix3Xd to_hats(Matrix3d transf) {
    const auto visitor = overloads{
        [this, transf](Eigen::Matrix<f64, 3, 6>) {
          std::cout << "found hexagon\n";
          return Matrix3Xd{
              transf * transform *
              (Matrix3Xd(3, 4 * 13)
                   << translate_by(sixth_rot(5), {2.5, 0.5 * sq3}) * hat1,
               translate_by(sixth_rot(4), {10., sq3}) * hat,
               translate_by(sixth_rot(4), {4., sq3}) * hat,
               translate_by(sixth_rot(2), {2.5, 1.5 * sq3}) * hat)
                  .finished()};
        },
        [this, transf](Eigen::Matrix<f64, 3, 3>) {
          std::cout << "Found triangle\n";
          std::cout << "Should output transformed version of:\n" << hat << '\n';
          return Matrix3Xd(transf * transform * transl2({0.5, 0.5 * sq3}) *
                           hat);
        },
        [this, transf](Eigen::Matrix<f64, 3, 4>) {
          std::cout << "Found parallellogram\n";
          return Matrix3Xd{
              transf * transform *
              (Matrix3Xd(3, 2 * 13) << transl2({1.5, 0.5 * sq3}) * hat,
               translate_by(sixth_rot(5), {0., sq3}) * hat)
                  .finished()};
        },
        [this, transf](Eigen::Matrix<f64, 3, 5>) {
          std::cout << "Found pentagon\n";
          return Matrix3Xd{
              transf * transform *
              (Matrix3Xd(3, 2 * 13) << transl2({1.5, 0.5 * sq3}) * hat,
               translate_by(sixth_rot(5), {0., sq3}) * hat)
                  .finished()};
        }};
    return std::visit<Matrix3Xd>(visitor, *p_shape); //->visit(visitor);
    // return hat;
  }
};

int main(int argc, char* argv[]) {
  cxxopts::Options options("test program", "bleh");
  options.add_options()("v,verbose", "Verbose output", cxxopts::value<bool>());
  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
  if (result["v"].count()) {
    spdlog::set_level(spdlog::level::trace);
  }

  ShapeMaker bleh{Matrix3d::Identity(),
                  std::make_shared<shape_var_t>(shape_var_t{
                      Matrix3d{{0., 0., 0.}, {0., 0., 0.}, {1., 1., 1.}}})};
  auto hex = std::make_shared<shape_var_t>(
      shape_var_t{(Eigen::Matrix<f64, 3, 6>() << 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 1., 1., 1., 1., 1., 1.)
                      .finished()});
  ShapeMaker blah{Matrix3d::Identity(), hex};
  auto para = std::make_shared<shape_var_t>(
      shape_var_t{(Eigen::Matrix<f64, 3, 4>() << 0., 0., 0., 0., 0., 0., 0., 0.,
                   1., 1., 1., 1.)
                      .finished()});
  ShapeMaker bloh{Matrix3d::Identity(), para};
  auto penta = std::make_shared<shape_var_t>(
      shape_var_t{(Eigen::Matrix<f64, 3, 5>() << 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 1., 1., 1., 1., 1.)
                      .finished()});
  ShapeMaker bluh{Matrix3d::Identity(), penta};
  std::cout << bleh.to_hats(Matrix3d::Identity()) << '\n';
  std::cout << blah.to_hats(Matrix3d::Identity()) << '\n';
  std::cout << bloh.to_hats(Matrix3d::Identity()) << '\n';
  std::cout << bluh.to_hats(Matrix3d::Identity()) << '\n';

  return 0;
}
