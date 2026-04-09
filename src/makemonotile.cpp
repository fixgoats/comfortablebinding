#include "Eigen/Core"
#include "Eigen/Dense"
#include "betterexc.h"
#include "kdtree.h"
#include "raylib.h"
#include "typedefs.h"
#include <cmath>
#include <cstddef>
#include <exception>
#include <vector>

using Eigen::Vector2d, Eigen::Matrix3d, Eigen::Matrix3Xd, Eigen::Vector3d,
    Eigen::Matrix2d, Eigen::indexing::all, Eigen::MatrixX3d;

constexpr double sq3 = 1.7320508075688772935;

static const Matrix3Xd hat =
    (Matrix3Xd(3, 13) << 0., -0.75, -0.5, 0.5, 0.75, 1.5, 2.25, 2., 1.5, 1.5,
     0.75, 0.5, 0., 0., -0.25 * sq3, -0.5 * sq3, -0.5 * sq3, -0.25 * sq3,
     -0.5 * sq3, -0.25 * sq3, 0., 0., 0.5 * sq3, 0.75 * sq3, 0.5 * sq3,
     0.5 * sq3, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.)
        .finished();
// c64{0., 0.}, c64{-0.75, -0.25 * sq3}, c64{-0.5, -0.5 * sq3},
//     c64{0.5, -0.5 * sq3}, c64{0.75, -0.25 * sq3}, c64{1.5, -0.5 * sq3},
//     c64{2.25, -0.25 * sq3}, c64{2., 0.}, c64{1.5, 0.}, c64{1.5, 0.5 * sq3},
//     c64{0.75, 0.75 * sq3}, c64{0.5, 0.5 * sq3}, c64 {
//   0., 0.5 * sq3
// }

static const Matrix3Xd hat1 =
    (Eigen::Matrix3Xd(3, 13) << 0., 0.75, 0.5, -0.5, -0.75, -1.5, -2.25, -2.,
     -1.5, -1.5, -0.75, -0.5, 0., 0., -0.25 * sq3, -0.5 * sq3, -0.5 * sq3,
     -0.25 * sq3, -0.5 * sq3, -0.25 * sq3, 0., 0., 0.5 * sq3, 0.75 * sq3,
     0.5 * sq3, 0.5 * sq3, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.)
        .finished();

enum MetaType {
  H,
  T,
  P,
  F,
};

struct HatTile {
  Matrix3d transform;
  bool mirrored;

  constexpr Matrix3Xd points() const {
    if (mirrored) {
      return transform * hat1;
    }
    return transform * hat;
  }
};

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
    return (Matrix3d() << -1.0, 0., 0., 0., -1., 0., 0., 0., 1.).finished();
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
  return (Matrix3d() << 0., 0., v[0], 0., 0., v[1], 0., 0., 1.).finished();
}

constexpr Matrix3d transl3(Vector3d v) {
  return (Matrix3d() << 0., 0., v[0], 0., 0., v[1], 0., 0., 1.).finished();
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

struct MetaTile {
  Eigen::Matrix3Xd shape;
  Matrix3d transform;
  MetaType type;
  std::vector<MetaTile*> children;

  void hats_recursive(Matrix3d trans, std::vector<HatTile>& hat_vec) const {
    if (children.size() == 0) {
      switch (type) {
      case H:
        hat_vec.insert(
            hat_vec.end(),
            {{trans * translate_by(sixth_rot(5), {2.5, 0.5 * sq3}), true},
             {trans * translate_by(sixth_rot(4), {10., sq3}), false},
             {trans * translate_by(sixth_rot(4), {4., sq3}), false},
             {trans * translate_by(sixth_rot(2), {2.5, 1.5 * sq3}), false}});
      case T:
        hat_vec.insert(hat_vec.end(),
                       {{trans * transl2({0.5, 0.5 * sq3}), false}});
      case P:
      case F:
        hat_vec.insert(
            hat_vec.end(),
            {{trans * transl2({1.5, 0.5 * sq3}), false},
             {trans * translate_by(sixth_rot(5), {0., sq3}), false}});
      }
    } else {
      for (const auto& ch : children) {
        hats_recursive(trans * ch->transform, hat_vec);
      }
    }
  }
  std::vector<HatTile> hats() const {
    std::vector<HatTile> hats;
    hats.reserve(10000);
    hats_recursive(transform, hats);
    return hats;
  }

  constexpr Vector3d eval_child(size_t i, size_t j) const {
    return children[i]->transform * children[i]->shape(all, j);
  }

  Matrix3Xd to_points() const {
    const auto hat_vec = hats();
    Matrix3Xd ret(3, hat_vec.size() * 13);
    for (u64 i = 0; i < hat_vec.size(); ++i) {
      ret(all, Eigen::seq(i * 13, (i + 1) * 13 - 1)) = hat_vec[i].points();
    }
    return ret;
  }

  std::vector<Matrix3Xd> to_polys() const {
    const auto hat_vec = hats();
    std::vector<Matrix3Xd> ret_vec(hat_vec.size());
    for (u64 i = 0; i < hat_vec.size(); ++i) {
      ret_vec[i] = hat_vec[i].points();
    }

    return ret_vec;
  }

  size_t shape_len() const {
    switch (type) {
    case H:
      return 6;
    case T:
      return 3;
    case P:
      return 4;
    case F:
      return 5;
    }
  }
  void clearRecursive(MetaTile* mt) {
    if (mt == nullptr)
      return;
    for (auto ch : mt->children) {
      clearRecursive(ch);
    }
    delete mt;
  }
  void clear() {
    for (auto ch : children) {
      clearRecursive(ch);
    }
    children.clear();
  }
  ~MetaTile() { clear(); }
};

constexpr Vector3d affsub(Vector3d v, Vector3d u) {
  return {v[0] - u[0], v[1] - u[1], 1.};
}

constexpr Vector3d affadd(Vector3d v, Vector3d u) {
  return {v[0] + u[0], v[1] + u[1], 1.};
}

static const std::vector<std::vector<size_t>> rules{{0},
                                                    {2, 0, 0, 2},
                                                    {0, 1, 0, 2},
                                                    {2, 2, 0, 2},
                                                    {0, 3, 0, 2},
                                                    {2, 4, 4, 2},
                                                    {3, 0, 4, 3},
                                                    {3, 2, 4, 3},
                                                    {3, 4, 1, 3, 2, 0},
                                                    {0, 8, 3, 0},
                                                    {2, 9, 2, 0},
                                                    {0, 10, 2, 0},
                                                    {2, 11, 4, 2},
                                                    {0, 12, 0, 2},
                                                    {3, 13, 0, 3},
                                                    {3, 14, 2, 1},
                                                    {0, 15, 3, 4},
                                                    {3, 8, 2, 1},
                                                    {0, 17, 3, 0},
                                                    {2, 18, 2, 0},
                                                    {0, 19, 2, 2},
                                                    {3, 20, 4, 3},
                                                    {2, 20, 0, 2},
                                                    {0, 22, 0, 2},
                                                    {3, 23, 4, 3},
                                                    {3, 23, 0, 3},
                                                    {2, 16, 0, 2},
                                                    {1, 9, 4, 0, 2, 2},
                                                    {3, 4, 0, 3}};

MetaTile construct_patch(MetaTile* shapes) {
  MetaTile ret;
  ret.children.reserve(29);
  for (const auto& r : rules) {
    if (r.size() == 1) {
      MetaTile* nshp = new MetaTile(shapes[0]);
      ret.children.push_back(nshp);
    } else if (r.size() == 4) {
      // const Eigen::Matrix3Xd poly = ret.children[r[1]]->shape;
      const Vector3d p =
          ret.eval_child(r[1], (r[2] + 1) % ret.children[r[1]]->shape_len());
      const Vector3d q = ret.eval_child(r[1], r[2]);
      MetaTile* nshp = new MetaTile(shapes[r[0]]);
      const auto match_transf =
          match_segs(nshp->shape(all, r[3]),
                     nshp->shape(all, (r[3] + 1) % nshp->shape_len()), p, q);
      nshp->transform = match_transf;
      ret.children.push_back(nshp);
    } else {
      const Vector3d p = ret.eval_child(r[1], r[4]);
      const Vector3d q = ret.eval_child(r[3], r[2]);
      MetaTile* nshp = new MetaTile(shapes[r[0]]);
      const auto match_trans =
          match_segs(nshp->shape(all, r[5]),
                     nshp->shape(all, (r[5] + 1) % nshp->shape_len()), p, q);
      nshp->transform = match_trans;
      ret.children.push_back(nshp);
    }
  }
  return ret;
}

MetaTile* construct_metatiles(const MetaTile& patch) {
  MetaTile* ret = new MetaTile[4];
  const auto bps1 = patch.eval_child(8, 2);
  const auto bps2 = patch.eval_child(21, 2);
  const auto rbps = rot_about(bps1, 4) * bps2;
  const auto p72 = patch.eval_child(7, 2);
  const auto p252 = patch.eval_child(25, 2);
  const auto llc = intersection(bps1, rbps, patch.eval_child(6, 2), p72);

  auto w = affsub(patch.eval_child(6, 2), llc);

  Matrix3Xd new_h_outline(3, 6);
  new_h_outline(all, 0) = llc;
  new_h_outline(all, 1) = bps1;
  w = sixth_rot(5) * w;
  new_h_outline(all, 2) = affadd(new_h_outline(all, 1), w);
  new_h_outline(all, 3) = patch.eval_child(14, 2);
  w = sixth_rot(5) * w;
  new_h_outline(all, 4) = affsub(new_h_outline(all, 3), w);
  new_h_outline(all, 5) = patch.eval_child(6, 2);
  ret[0] = MetaTile{new_h_outline,
                    Matrix3d::Identity(),
                    MetaType::H,
                    {patch.children[0], patch.children[9], patch.children[16],
                     patch.children[27], patch.children[26], patch.children[6],
                     patch.children[1], patch.children[8], patch.children[10],
                     patch.children[15]}};

  const Vector3d aaa = new_h_outline(all, 2);
  const Vector3d bbb =
      affadd(new_h_outline(all, 1),
             affsub(new_h_outline(all, 4), new_h_outline(all, 5)));
  const Vector3d ccc = rot_about(bbb, 5) * aaa;
  ret[1] = MetaTile{(Matrix3Xd(3, 3) << bbb, ccc, aaa).finished(),
                    Matrix3d::Identity(),
                    MetaType::T,
                    {
                        patch.children[11],
                    }

  };

  ret[2] = MetaTile{
      (Matrix3Xd(3, 4) << p72, affadd(p72, affsub(bps1, llc)), bps1, llc)
          .finished(),
      Matrix3d::Identity(),
      MetaType::P,
      {
          patch.children[7],
          patch.children[2],
          patch.children[3],
          patch.children[4],
          patch.children[28],
      }

  };

  ret[3] =
      MetaTile{(Matrix3Xd(3, 5) << bps2, patch.eval_child(24, 2),
                patch.eval_child(25, 0), p252, affadd(p252, affsub(llc, bps1)))
                   .finished(),
               Matrix3d::Identity(),
               MetaType::P,
               {patch.children[21], patch.children[20], patch.children[22],
                patch.children[23], patch.children[24], patch.children[25]}};
  return ret;
}

s32 toScreen(f64 r, f64 min, f64 max, s32 dim) {
  return (s32)(((r - min) / (max - min)) * (f64)dim);
}

int main(int argc, char* argv[]) {
  auto init_h = MetaTile{(Matrix3Xd(3, 6) << 0., 4., 4.5, 2.5, 1.5, -0.5, 0.,
                          0., 0.5 * sq3, 2.5 * sq3, 2.5 * sq3, 0.5 * sq3, 1.,
                          1., 1., 1., 1., 1.)
                             .finished(),
                         Matrix3d::Identity(),
                         MetaType::H,
                         {}};
  auto init_t =
      MetaTile{(Matrix3Xd(3, 3) << 0., 3., 1.5, 0., 0., 1.5 * sq3, 1., 1., 1.)
                   .finished(),
               Matrix3d::Identity(),
               MetaType::T,
               {}};
  auto init_p = MetaTile{
      (Matrix3Xd(3, 4) << 0., 4., 3., -1., 0., 0., sq3, sq3, 1., 1., 1., 1.)
          .finished(),
      Matrix3d::Identity(),
      MetaType::P,
      {}};
  auto init_f = MetaTile{(Matrix3Xd(3, 4) << 0., 3., 3.5, 3., -1., 0., 0.,
                          0.5 * sq3, sq3, sq3, 1., 1., 1., 1., 1.)
                             .finished(),
                         Matrix3d::Identity(),
                         MetaType::F,
                         {}};

  s32 width = 800;
  s32 height = 800;
  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE |
                 FLAG_WINDOW_TRANSPARENT);
  InitWindow(width, height, "raylib test");

  SetTargetFPS(10);
  const auto points = init_h.to_points();                 // hat;
  const std::vector<Matrix3Xd> polys = init_h.to_polys(); // {hat};
  f64 xmin = points(0, all).minCoeff();
  f64 xmax = points(0, all).maxCoeff();
  f64 ymin = points(1, all).minCoeff();
  f64 ymax = points(1, all).maxCoeff();
  double exmin = xmin - 0.05 * (xmax - xmin);
  double eymin = ymin - 0.05 * (ymax - ymin);
  double exmax = xmax + 0.05 * (xmax - xmin);
  double eymax = ymax + 0.05 * (ymax - ymin);
  while (!WindowShouldClose()) {
    width = GetScreenWidth();
    height = GetScreenHeight();
    BeginDrawing();
    ClearBackground(WHITE);
    for (s32 i = 0; i < points.cols(); ++i) {
      DrawCircle(toScreen(points(0, i), exmin, exmax, width),
                 toScreen(points(1, i), eymin, eymax, height), 12.0, BLACK);
    }
    for (u32 i = 0; i < polys.size(); ++i) {
      for (s32 j = 0; j < polys[i].cols(); ++j) {
        DrawLine(toScreen(polys[i](0, j), exmin, exmax, width),
                 toScreen(polys[i](1, j), eymin, eymax, height),
                 toScreen(polys[i](0, (j + 1) % polys[i].cols()), exmin, exmax,
                          width),
                 toScreen(polys[i](1, (j + 1) % polys[i].cols()), eymin, eymax,
                          height),
                 BLACK);
      }
    }
    EndDrawing();
  }
  CloseWindow();
  return 0;
}
