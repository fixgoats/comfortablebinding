#include "SDF.h"
#include "Eigen/Dense"
#include "io.h"
// #include "vkcore.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include <chrono>
#include <iostream>
#include <mdspan>
#include <toml++/toml.hpp>

using namespace std::chrono;
using Eigen::MatrixXcf;

namespace {
void auto_limits(const VectorXd& d, RangeConf<f64>& rc) {
  f64 max = d.maxCoeff();
  f64 min = d.minCoeff();
  f64 l = max - min;
  rc.start = min - 0.01 * l;
  rc.end = max + 0.01 * l;
}

// VectorXcd plane_wave(Vector2d k, const std::vector<Pt2>& points) {
//   VectorXcd tmp = VectorXcd::Zero(static_cast<s64>(points.size()));
//   std::ranges::transform(points.begin(), points.end(), tmp.begin(), [&](Pt2
//   p) {
//     return (1. / sqrt(points.size())) *
//            std::exp(c64{0, k(0) * p[0] + k(1) * p[1]});
//   });
//   return VectorXcd{tmp.transpose()};
// }

} // namespace

// VectorXcd plane_wave(Vector2d k, const std::vector<Point>& points) {
//   VectorXcd tmp = VectorXcd::Zero(static_cast<s64>(points.size()));
//   std::ranges::transform(points.begin(), points.end(), tmp.begin(),
//                          [&](Point p) {
//                            return (1. / sqrt(points.size())) *
//                                   std::exp(c64{0, k(0) * p[0] + k(1) *
//                                   p[1]});
//                          });
//   return VectorXcd{tmp.transpose()};
// }

Delta delta(const VectorXd& d, RangeConf<f64> ec, f64 sharpening, f64 cutoff) {
  const f64 nonzero_range = std::sqrt(-std::log(cutoff) / sharpening);
  Delta delta(ec.n);
  for (u32 i = 0; i < ec.n; i++) {
    const f64 e = ec.ith(i);
    delta[i] = [&]() {
      std::vector<std::pair<f64, u32>> tmp;
      tmp.reserve(5);
      for (u32 k = 0; k < d.size(); k++) {
        if (f64 diff = std::abs(d(k) - e); diff < nonzero_range) {
          tmp.emplace_back(std::exp(-sharpening * square(diff)), k);
        }
      }
      tmp.shrink_to_fit();
      return tmp;
    }();
  }
  return delta;
}

Delta delta(const VectorXcd& d, RangeConf<f64> ec, f64 sharpening, f64 cutoff) {
  const f64 nonzero_range = std::sqrt(-std::log(cutoff) / sharpening);
  Delta delta(ec.n);
  for (u32 i = 0; i < ec.n; i++) {
    const f64 e = ec.ith(i);
    delta[i] = [&]() {
      std::vector<std::pair<f64, u32>> tmp;
      tmp.reserve(5);
      for (u32 k = 0; k < d.size(); k++) {
        if (f64 diff = std::abs(d(k).real() - e); diff < nonzero_range) {
          tmp.emplace_back(std::exp(-sharpening * square(diff)), k);
        }
      }
      tmp.shrink_to_fit();
      return tmp;
    }();
  }
  return delta;
}

// std::vector<f64> full_sdf(const VectorXd& d, const MatrixXcd& uh,
//                           const std::vector<Point>& points, f64 lat_const,
//                           RangeConf<f64> kxc, RangeConf<f64> kyc,
//                           RangeConf<f64>& ec, f64 sharpening, f64 cutoff,
//                           bool print_progress) {
//   std::cout << "Calculating full SDF\n";
//   const u32 its = kxc.n / 10;
//   std::vector<f64> sdf(ec.n * kyc.n * kxc.n, 0);
//   auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n, ec.n);
//   if (fleq(ec.start, ec.end, 1e-16)) {
//     auto_limits(d, ec);
//   }
//   auto del = delta(d, ec, sharpening, cutoff);
//   if (print_progress) {
//     std::cout << "[" << std::flush;
//   }
// #pragma omp parallel for
//   for (size_t i = 0; i < kxc.n; i++) {
//     const f64 kx = kxc.ith(i) * 2 * M_PI / lat_const;
//     for (u64 j = 0; j < kyc.n; j++) {
//       const f64 ky = kyc.ith(j) * 2 * M_PI / lat_const;
//       const VectorXcd k_vec = uh * plane_wave({kx, ky}, points);
//       for (u32 k = 0; k < ec.n; k++) {
//         for (const auto& pair : del[k]) {
//           sdf_view[i, j, k] += pair.first * std::norm(k_vec(pair.second));
//         }
//       }
//     }
//     if (print_progress) {
//       if (i % its == 0) {
//         std::cout << "█|" << std::flush;
//       }
//     }
//   }
// #pragma omp barrier
//   if (print_progress) {
//     std::cout << "█]\n";
//   }
//   return sdf;
// }

// MatrixXd non_herm_disp(const VectorXcd& d, const MatrixXcd& uh,
//                        const std::vector<Point>& points, f64 lat_const,
//                        RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
//                        f64 sharpening, f64 cutoff, bool print_progress) {
//   std::cout << "Lattice constant (sort of) is: " << lat_const << '\n';
//   std::cout << "Calculating cross section of SDF at E = " << e << '\n';
//   const u32 its = kxc.n / 10;
//   MatrixXd sdf(kyc.n, kxc.n);
//   // auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
//   auto del = delta(d, {.start = e, .end = e, .n = 1}, sharpening, cutoff);
//   if (print_progress) {
//     std::cout << "[" << std::flush;
//   }
//   const std::vector<size_t> indices = [&]() {
//     std::vector<size_t> tmp;
//     for (const auto pair : del[0]) {
//       tmp.push_back(pair.second);
//     }
//     return tmp;
//   }();
//   MatrixXcd restricted_uh = uh(indices, Eigen::indexing::all);
// #pragma omp parallel for
//   for (size_t i = 0; i < kxc.n; i++) {
//     const f64 kx = kxc.ith(i) * 2 * M_PI / lat_const;
//     std::vector<u64> cur_element(kyc.n, 0);
//     for (u64 j = 0; j < kyc.n; j++) {
//       const f64 ky = kyc.ith(j) * 2 * M_PI / lat_const;
//       const VectorXcd k_vec = restricted_uh * plane_wave({kx, ky}, points);
//       u32 cur_element = 0;
//       for (const auto& pair : del[0]) {
//         sdf(static_cast<s64>(j), static_cast<s64>(i)) +=
//             pair.first * std::norm(k_vec(cur_element));
//         cur_element += 1;
//       }
//     }
//     if (print_progress) {
//       if (i % its == 0) {
//         std::cout << "█|" << std::flush;
//       }
//     }
//   }
// #pragma omp barrier
//   if (print_progress) {
//     std::cout << "█]\n";
//   }
//   return sdf;
// }

// MatrixXd non_herm_e_section(const VectorXd& d, const MatrixXcd& uh,
//                             const std::vector<Point>& points, f64 lat_const,
//                             RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
//                             f64 sharpening, f64 cutoff, bool print_progress)
//                             {
//   std::cout << "Lattice constant (sort of) is: " << lat_const << '\n';
//   std::cout << "Calculating cross section of SDF at E = " << e << '\n';
//   const u32 its = kxc.n / 10;
//   MatrixXd sdf(kyc.n, kxc.n);
//   // auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
//   auto del = delta(d, {.start = e, .end = e, .n = 1}, sharpening, cutoff);
//   if (print_progress) {
//     std::cout << "[" << std::flush;
//   }
//   const std::vector<size_t> indices = [&]() {
//     std::vector<size_t> tmp;
//     for (const auto pair : del[0]) {
//       tmp.push_back(pair.second);
//     }
//     return tmp;
//   }();
//   MatrixXcd restricted_uh = uh(indices, Eigen::indexing::all);
// #pragma omp parallel for
//   for (size_t i = 0; i < kxc.n; i++) {
//     const f64 kx = kxc.ith(i) * 2 * M_PI / lat_const;
//     std::vector<u64> cur_element(kyc.n, 0);
//     for (u64 j = 0; j < kyc.n; j++) {
//       const f64 ky = kyc.ith(j) * 2 * M_PI / lat_const;
//       const VectorXcd k_vec = restricted_uh * plane_wave({kx, ky}, points);
//       u32 cur_element = 0;
//       for (const auto& pair : del[0]) {
//         sdf(static_cast<s64>(j), static_cast<s64>(i)) +=
//             pair.first * std::norm(k_vec(cur_element));
//         cur_element += 1;
//       }
//     }
//     if (print_progress) {
//       if (i % its == 0) {
//         std::cout << "█|" << std::flush;
//       }
//     }
//   }
// #pragma omp barrier
//   if (print_progress) {
//     std::cout << "█]\n";
//   }
//   return sdf;
// }
//
// MatrixXd e_section(const VectorXd& d, const MatrixXcd& uh,
//                    const std::vector<Point>& points, f64 lat_const,
//                    RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
//                    f64 sharpening, f64 cutoff, bool print_progress) {
//   std::cout << "Lattice constant (sort of) is: " << lat_const << '\n';
//   std::cout << "Calculating cross section of SDF at E = " << e << '\n';
//   const u32 its = kxc.n / 10;
//   MatrixXd sdf(kyc.n, kxc.n);
//   // auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
//   auto del = delta(d, {.start = e, .end = e, .n = 1}, sharpening, cutoff);
//   if (print_progress) {
//     std::cout << "[" << std::flush;
//   }
//   const std::vector<size_t> indices = [&]() {
//     std::vector<size_t> tmp;
//     for (const auto pair : del[0]) {
//       tmp.push_back(pair.second);
//     }
//     return tmp;
//   }();
//   MatrixXcd restricted_uh = uh(indices, Eigen::indexing::all);
// #pragma omp parallel for
//   for (size_t i = 0; i < kxc.n; i++) {
//     const f64 kx = kxc.ith(i) * 2 * M_PI / lat_const;
//     std::vector<u64> cur_element(kyc.n, 0);
//     for (u64 j = 0; j < kyc.n; j++) {
//       const f64 ky = kyc.ith(j) * 2 * M_PI / lat_const;
//       const VectorXcd k_vec = restricted_uh * plane_wave({kx, ky}, points);
//       u32 cur_element = 0;
//       for (const auto& pair : del[0]) {
//         sdf(static_cast<s64>(j), static_cast<s64>(i)) +=
//             pair.first * std::norm(k_vec(cur_element));
//         cur_element += 1;
//       }
//     }
//     if (print_progress) {
//       if (i % its == 0) {
//         std::cout << "█|" << std::flush;
//       }
//     }
//   }
// #pragma omp barrier
//   if (print_progress) {
//     std::cout << "█]\n";
//   }
//   return sdf;
// }
//
// std::vector<f64> dos(const VectorXd& d, const MatrixXcd& uh,
//                      const std::vector<Point>& points, f64 lat_const,
//                      RangeConf<f64> kxc, RangeConf<f64> kyc, RangeConf<f64>&
//                      ec, f64 sharpening, f64 cutoff, bool print_progress) {
//   std::cout << "Calculating density of states\n";
//   const u32 its = ec.n / 10;
//   std::vector<f64> densities(ec.n, 0);
//   if (fleq(ec.start, ec.end, 1e-16)) {
//     auto_limits(d, ec);
//   }
//   auto del = delta(d, ec, sharpening, cutoff);
//   if (print_progress) {
//     std::cout << "[" << std::flush;
//   }
//   for (u32 k = 0; k < ec.n; k++) {
//     std::vector<f64> pre_densities(kxc.n * kyc.n, 0);
//     auto pre_density_view = std::mdspan(pre_densities.data(), kxc.n, kyc.n);
//     for (const auto& pair : del[k]) {
// #pragma omp parallel for
//       for (u32 i = 0; i < kxc.n; i++) {
//         const f64 kx = kxc.ith(i) * 2 * M_PI / lat_const;
//         for (u32 j = 0; j < kyc.n; j++) {
//           const f64 ky = kyc.ith(j) * 2 * M_PI / lat_const;
//           const VectorXcd k_vec = uh * plane_wave({kx, ky}, points);
//           pre_density_view[i, j] += pair.first *
//           std::norm(k_vec(pair.second));
//         }
//       }
// #pragma omp barrier
//     }
//     for (u32 i = 0; i < kxc.n; i++) {
//       for (u32 j = 0; j < kyc.n; j++) {
//         densities[k] += pre_density_view[i, j];
//       }
//     }
//     if (print_progress) {
//       if (k % its == 0) {
//         std::cout << "█|" << std::flush;
//       }
//     }
//   }
//   if (print_progress) {
//     std::cout << "█]\n";
//   }
//   return densities;
// }
//
// MatrixXd disp(const VectorXd& d, const MatrixXcd& uh,
//               const std::vector<Point>& points, f64 lat_const,
//               std::vector<RangeConf<Vector2d>> kc, RangeConf<f64>& ec,
//               f64 sharpening, f64 cutoff, bool print_progress) {
//   std::cout << "Calculating dispersion relation\n";
//   u32 nsamples = 0;
//   for (const auto& r : kc) {
//     nsamples += r.n;
//   }
//   const u32 its = nsamples / 10;
//   std::cout << "its: " << its << '\n';
//   MatrixXd disp(ec.n, nsamples);
//   /// std::vector<f64> disp(nsamples * ec.n, 0);
//   if (fleq(ec.start, ec.end, 1e-16)) {
//     auto_limits(d, ec);
//   }
//   auto del = delta(d, ec, sharpening, cutoff);
//   auto disp_view = std::mdspan(disp.data(), ec.n, nsamples);
//   if (print_progress) {
//     std::cout << "[" << std::flush;
//   }
//   // duration<u64, nanoseconds::period> kvecduration{};
//   // duration<u64, nanoseconds::period> dispcalcduration{};
//   u32 it = 0;
//   u32 count = 0;
//   for (const auto& rc : kc) {
//     for (u32 i = 0; i < rc.n; i++) {
//       const auto k = rc.ith(i);
//       // auto start = steady_clock::now();
//       const VectorXcd k_vec = uh * plane_wave(k, points);
//       // auto end = steady_clock::now();
//       // kvecduration += end - start;
// #pragma omp parallel for
//       for (u32 j = 0; j < ec.n; j++) {
//         // auto sumstart = steady_clock::now();
//         for (const auto& pair : del[j]) {
//           disp(static_cast<s64>(j), static_cast<s64>(it) + i) +=
//               pair.first * std::norm(k_vec(pair.second));
//         }
//         // auto sumend = steady_clock::now();
//         // dispcalcduration += sumend - sumstart;
//       }
// #pragma omp barrier
//       ++count;
//       if (print_progress) {
//         if (count % its == 0) {
//           std::cout << "█|" << std::flush;
//         }
//       }
//     }
//     it += rc.n;
//     std::cout << count << '\n';
//   }
//   if (print_progress) {
//     std::cout << "█]\n";
//   }
//   // std::cout << "Time to initialize k states: " << kvecduration << '\n';
//   // std::cout << "Time to sum disp values: " << dispcalcduration << '\n';
//   return disp;
// }
//
// MatrixXd points_to_finite_hamiltonian(const std::vector<Point>& points,
//                                       const kdt::KDTree<Point>& kdtree,
//                                       f64 radius) {
//   /* This function creates a hamiltonian for a simple finite lattice.
//    * Can't exactly do a dispersion from this.
//    */
//   std::vector<Neighbour> nb_info;
//   for (size_t i = 0; i < points.size(); i++) {
//     auto q = points[i];
//     auto nbs = kdtree.radiusSearch(q, radius);
//     for (const auto idx : nbs) {
//       if ((size_t)idx > i) {
//         auto p = points[idx];
//         Vector2d d = {p[0] - q[0], p[1] - q[1]};
//         nb_info.emplace_back(i, p.idx, d);
//       }
//     }
//   }
//   return finite_hamiltonian(points.size(), nb_info, &expCoupling);
// }
//
// #define SET_STRUCT_FIELD(c, tbl, key) \
//   if ((tbl).contains(#key)) \ (c).key =
//   *(tbl)[#key].value<decltype((c).key)>()
//
// std::optional<SdfConf> toml_to_sdf_conf(const std::string& toml_path) {
//   toml::table tbl;
//   try {
//     tbl = toml::parse_file(toml_path);
//   } catch (const std::exception& err) {
//     std::cerr << "Parsing file " << toml_path
//               << " failed with exception: " << err.what() << '\n';
//     return {};
//   }
//   SdfConf conf{};
//   toml::table& pre_conf = *tbl["PreConf"].as_table();
//   SET_STRUCT_FIELD(conf, pre_conf, sharpening);
//   SET_STRUCT_FIELD(conf, pre_conf, cutoff);
//   SET_STRUCT_FIELD(conf, pre_conf, search_radius);
//   SET_STRUCT_FIELD(conf, pre_conf, point_path);
//   SET_STRUCT_FIELD(conf, pre_conf, fname_h5);
//   SET_STRUCT_FIELD(conf, pre_conf, do_full_sdf);
//   SET_STRUCT_FIELD(conf, pre_conf, do_dos);
//   SET_STRUCT_FIELD(conf, pre_conf, do_e_section);
//   SET_STRUCT_FIELD(conf, pre_conf, do_path);
//   SET_STRUCT_FIELD(conf, pre_conf, fixed_e);
//   if (conf.do_path) {
//     try {
//       toml::table path_spec = *tbl["path"].as_table();
//       toml::array ranges = *path_spec["ranges"].as_array();
//       for (const auto& range : ranges) {
//         toml::table r = *range.as_table();
//         conf.kpath.push_back(tblToVecRange(r));
//       }
//       conf.disp_e = tblToRange(*tbl["dispE"].as_table());
//     } catch (const std::exception& exc) {
//       std::cerr << "Path dispersion requested but no ranges specified: "
//                 << exc.what() << std::endl;
//     }
//   }
//
//   if (pre_conf.contains("save_diag")) {
//     conf.save_diag = pre_conf["save_diag"].value<std::string>();
//   }
//   if (pre_conf.contains("use_saved_diag")) {
//     conf.use_saved_diag = pre_conf["use_saved_diag"].value<std::string>();
//   }
//   if (pre_conf.contains("save_hamiltonian")) {
//     conf.save_hamiltonian =
//     pre_conf["save_hamiltonian"].value<std::string>();
//   }
//   conf.section_kx = tblToRange(*tbl["section_kx"].as_table());
//   conf.section_ky = tblToRange(*tbl["section_ky"].as_table());
//   conf.sdf_kx = tblToRange(*tbl["sdf_kx"].as_table());
//   conf.sdf_ky = tblToRange(*tbl["sdf_ky"].as_table());
//   conf.sdf_e = tblToRange(*tbl["sdf_e"].as_table());
//   return conf;
// }
// #undef SET_STRUCT_FIELD
//
// int do_sdf_calcs(SdfConf& conf) {
//   if (!(conf.do_path || conf.do_e_section || conf.do_dos ||
//   conf.do_full_sdf)) {
//     std::cout << "No tasks selected.\n";
//     return 0;
//   }
//
//   std::vector<Point> points;
//   if (conf.point_path == "square") {
//     points.resize(70UL * 70UL);
//     for (u32 i = 0; i < 70; i++) {
//       for (u32 j = 0; j < 70; j++) {
//         points[i * 70 + j] = {(f64)i, (f64)j, i * 70 + j};
//       }
//     }
//   } else {
//     points = read_points(conf.point_path);
//   }
//
//   kdt::KDTree kdtree(points);
//   f64 a = avg_nn_dist(kdtree, points);
//   EigenSolution eigsol;
//   bool use_saved_succeeded = false;
//   if (conf.use_saved_diag.has_value()) {
//     const std::string& fname = conf.use_saved_diag.value();
//     if (file_exists(fname)) {
//       auto result = loadDiag(fname);
//       use_saved_succeeded = result.has_value();
//       if (use_saved_succeeded) {
//         eigsol = result.value();
//       }
//     }
//   }
//   if (!use_saved_succeeded) {
//     MatrixXd ham =
//         points_to_finite_hamiltonian(points, kdtree, conf.search_radius * a);
//     if (conf.save_hamiltonian.has_value()) {
//       saveEigen(conf.save_hamiltonian.value(), ham);
//     }
//     eigsol = hermitianEigenSolver(ham);
//   }
//   if (conf.save_diag.has_value() && !use_saved_succeeded) {
//     HighFive::File file(conf.save_diag.value(), HighFive::File::Truncate);
//     file.createDataSet("D", eigsol.D);
//     file.createDataSet("U", eigsol.U);
//   }
//   HighFive::File file(conf.fname_h5, HighFive::File::Truncate);
//   if (conf.do_e_section) {
//     auto uh = eigsol.U.adjoint();
//     auto section =
//         e_section(eigsol.D, uh, points, a, conf.section_kx, conf.section_ky,
//                   conf.fixed_e, conf.sharpening, conf.cutoff);
//     file.createDataSet("section", section);
//     std::array<f64, 5> sdf_bounds = {conf.section_kx.start,
//     conf.section_kx.end,
//                                      conf.section_ky.start,
//                                      conf.section_ky.end, conf.fixed_e};
//     file.createDataSet("section_bounds", sdf_bounds);
//   } else if (conf.do_full_sdf) {
//     auto uh = eigsol.U.adjoint();
//     auto sdf = full_sdf(eigsol.D, uh, points, a, conf.sdf_kx, conf.sdf_ky,
//                         conf.sdf_e, conf.sharpening, conf.cutoff);
//     file.createDataSet("sdf", sdf);
//     std::array<f64, 6> sdf_bounds = {conf.sdf_kx.start, conf.sdf_kx.end,
//                                      conf.sdf_ky.start, conf.sdf_ky.end,
//                                      conf.sdf_e.start,  conf.sdf_e.end};
//     file.createDataSet("sdf_bounds", sdf_bounds);
//   }
//   if (conf.do_dos) {
//     auto uh = eigsol.U.adjoint();
//     auto density_of_states =
//         dos(eigsol.D, uh, points, a, conf.sdf_kx, conf.sdf_ky, conf.sdf_e,
//             conf.sharpening, conf.cutoff);
//     file.createDataSet("dos", density_of_states);
//     std::array<f64, 2> dos_bounds = {conf.sdf_e.start, conf.sdf_e.end};
//     file.createDataSet("dos_bounds", dos_bounds);
//   }
//   if (conf.do_path) {
//     std::cout << "Doing path\n";
//     if (conf.disp_e.has_value()) {
//       auto kc = conf.kpath;
//       auto ec = conf.disp_e.value();
//       auto uh = eigsol.U.adjoint();
//       auto dis =
//           disp(eigsol.D, uh, points, a, kc, ec, conf.sharpening,
//           conf.cutoff);
//       u32 nsamples = 0;
//       for (const auto& rc : kc) {
//         nsamples += rc.n;
//       }
//       file.createDataSet("disp", dis);
//       std::vector<f64> disp_bounds;
//       for (const auto& rc : kc) {
//         disp_bounds.insert(disp_bounds.end(),
//                            {rc.start[0], rc.start[1], rc.end[0], rc.end[1]});
//       }
//       disp_bounds.insert(disp_bounds.end(), {ec.start, ec.end});
//       file.createDataSet("disp_bounds", disp_bounds);
//     } else {
//       std::cout << "Need to supply a non-zero number of energy samples\n";
//     }
//   }
//   return 0;
// }
