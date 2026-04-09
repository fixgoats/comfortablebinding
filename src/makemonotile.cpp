#include "Eigen/Dense"
#include "kdtree.h"
#include "typedefs.h"

using Eigen::VectorXcd, Eigen::Affine2d, Eigen::MatrixX2d;

constexpr double sq3 = 1.7320508075688772935;

static const Eigen::VectorXcd hat =
    (Eigen::VectorXcd() << c64{0., 0.}, c64{-0.75, -0.25 * sq3},
     c64{-0.5, -0.5 * sq3}, c64{0.5, -0.5 * sq3}, c64{0.75, -0.25 * sq3},
     c64{1.5, -0.5 * sq3}, c64{2.25, -0.25 * sq3}, c64{2., 0.}, c64{1.5, 0.},
     c64{1.5, 0.5 * sq3}, c64{0.75, 0.75 * sq3}, c64{0.5, 0.5 * sq3},
     c64{0., 0.5 * sq3})
        .finished();

static const Eigen::VectorXcd hat1 =
    (Eigen::VectorXcd() << c64{0., 0.}, c64{0.75, -0.25 * sq3},
     c64{0.5, -0.5 * sq3}, c64{-0.5, -0.5 * sq3}, c64{-0.75, -0.25 * sq3},
     c64{-1.5, -0.5 * sq3}, c64{-2.25, -0.25 * sq3}, c64{-2., 0.},
     c64{-1.5, 0.}, c64{-1.5, 0.5 * sq3}, c64{-0.75, 0.75 * sq3},
     c64{-0.5, 0.5 * sq3}, c64{0., 0.5 * sq3})
        .finished();

enum MetaType {
  H,
  T,
  P,
  F,
};

struct MetaTile {
  Eigen::MatrixX2d shape;
  Eigen::Affine2d transform;
  MetaType type;
  std::vector<std::unique_ptr<MetaTile>> children;

  Eigen::MatrixX2d hats_recursive(Affine2d trans) {
    if (children.size() == 0) {
      switch (type) {
      case H:
        Eigen::MatrixX2d bleh = trans * Eigen::AngleAxisd(M_PI, );
      }
    }
  }
};

int main(int argc, char* argv[]) { return 0; }
