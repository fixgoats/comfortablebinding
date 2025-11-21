#include "SDF.h"
#include "Eigen/Dense"
#include "io.h"
#include "vkcore.h"
#include <chrono>
#include <iostream>
#include <mdspan>
#include <toml++/toml.hpp>

using namespace std::chrono;
using Eigen::MatrixXcf;

void autoLimits(const VectorXd& D, RangeConf<f64>& rc) {
  f64 max = D.maxCoeff();
  f64 min = D.minCoeff();
  f64 l = max - min;
  rc.start = min - 0.01 * l;
  rc.end = max + 0.01 * l;
}

Delta delta(const VectorXd& D, RangeConf<f64> ec, f64 sharpening, f64 cutoff) {
  const f64 nonzero_range = std::sqrt(-std::log(cutoff) / sharpening);
  Delta delta(ec.n);
  for (u32 i = 0; i < ec.n; i++) {
    const f64 e = ec.ith(i);
    delta[i] = [&]() {
      std::vector<std::pair<f64, u32>> tmp;
      tmp.reserve(5);
      for (u32 k = 0; k < D.size(); k++) {
        if (f64 diff = std::abs(D(k) - e); diff < nonzero_range) {
          tmp.push_back({std::exp(-sharpening * square(diff)), k});
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

std::vector<f64> fullSDF(const VectorXd& D, const MatrixXcd& UH,
                         const std::vector<Point>& points, f64 lat_const,
                         RangeConf<f64> kxc, RangeConf<f64> kyc,
                         RangeConf<f64>& ec, f64 sharpening, f64 cutoff,
                         bool printProgress) {
  std::cout << "Calculating full SDF\n";
  const u32 its = kxc.n / 10;
  std::vector<f64> sdf(ec.n * kyc.n * kxc.n, 0);
  auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n, ec.n);
  if (fleq(ec.start, ec.end, 1e-16)) {
    autoLimits(D, ec);
  }
  auto del = delta(D, ec, sharpening, cutoff);
  if (printProgress)
    std::cout << "[" << std::flush;
#pragma omp parallel for
  for (size_t i = 0; i < kxc.n; i++) {
    const f64 kx = kxc.ith(i) * 2 * M_PI / lat_const;
    for (u64 j = 0; j < kyc.n; j++) {
      const f64 ky = kyc.ith(j) * 2 * M_PI / lat_const;
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

struct SpecConsts {
  u32 nx;
  u32 ny;
  u32 wx;
  u32 wy;
  u32 deltasize;
  f32 dk;
};

std::vector<f32> GPUEsection(Manager& m, const VectorXd& D, const MatrixXcd& UH,
                             const std::vector<Point>& points, f64 lat_const,
                             RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
                             f64 sharpening, f64 cutoff) {

  std::cout << "Calculating cross section of SDF at E = " << e << '\n';
  // const u32 its = kxc.n / 10;
  std::vector<f32> sdf(kyc.n * kxc.n, 0);
  auto del = delta(D, {e, e, 1}, sharpening, cutoff);
  const std::vector<size_t> indices = [&]() {
    std::vector<size_t> tmp;
    for (const auto pair : del[0]) {
      tmp.push_back(pair.second);
    }
    return tmp;
  }();
  const std::vector<f32> deltaCoeffs = [&]() {
    std::vector<f32> tmp;
    for (const auto c : del[0]) {
      tmp.push_back(c.first);
    }
    return tmp;
  }();
  std::vector<Vector2f> realpoints = [&]() {
    std::vector<Vector2f> tmp(points.size());
    for (u32 i = 0; i < points.size(); i++) {
      tmp[i] = points[i].asfVec();
    }
    return tmp;
  }();
  MatrixXcf floatUH = UH.cast<std::complex<f32>>();
  MatrixXcf restrictedUH = floatUH(indices, Eigen::indexing::all);
  SpecConsts sc{(u32)kxc.n, (u32)kyc.n,          32,
                32,         (u32)indices.size(), (f32)kxc.d()};
  auto gpuUH = m.makeRawBuffer<std::complex<f32>>(restrictedUH.size());
  auto gpuPoints = m.vecToBuffer(realpoints);
  auto gpuDelta = m.vecToBuffer(deltaCoeffs);
  auto gpuDensity = m.vecToBuffer(sdf);
  auto alg = m.makeAlgorithm("Shaders/esection.spv", {},
                             {&gpuUH, &gpuPoints, &gpuDelta, &gpuDensity}, sc);
  auto cb = m.beginRecord();
  appendOp(cb, alg, kxc.n / 32, kyc.n / 32, 1);
  cb.end();
  m.execute(cb);
  m.writeFromBuffer(gpuDensity, sdf);
  return sdf;
}

std::vector<f64> Esection(const VectorXd& D, const MatrixXcd& UH,
                          const std::vector<Point>& points, f64 lat_const,
                          RangeConf<f64> kxc, RangeConf<f64> kyc, f64 e,
                          f64 sharpening, f64 cutoff, bool printProgress) {
  std::cout << "Lattice constant (sort of) is: " << lat_const << '\n';
  std::cout << "Calculating cross section of SDF at E = " << e << '\n';
  const u32 its = kxc.n / 10;
  std::vector<f64> sdf(kyc.n * kxc.n, 0);
  auto sdf_view = std::mdspan(sdf.data(), kxc.n, kyc.n);
  auto del = delta(D, {e, e, 1}, sharpening, cutoff);
  if (printProgress)
    std::cout << "[" << std::flush;
  const std::vector<size_t> indices = [&]() {
    std::vector<size_t> tmp;
    for (const auto pair : del[0]) {
      tmp.push_back(pair.second);
    }
    return tmp;
  }();
  MatrixXcd restrictedUH = UH(indices, Eigen::indexing::all);
#pragma omp parallel for
  for (size_t i = 0; i < kxc.n; i++) {
    const f64 kx = kxc.ith(i) * 2 * M_PI / lat_const;
    std::vector<u64> cur_element(kyc.n, 0);
    for (u64 j = 0; j < kyc.n; j++) {
      const f64 ky = kyc.ith(j) * 2 * M_PI / lat_const;
      const VectorXcd k_vec = restrictedUH * planeWave({kx, ky}, points);
      u32 cur_element = 0;
      for (const auto& pair : del[0]) {
        sdf_view[i, j] += pair.first * std::norm(k_vec(cur_element));
        cur_element += 1;
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

std::vector<f64> DOS(const VectorXd& D, const MatrixXcd& UH,
                     const std::vector<Point>& points, f64 lat_const,
                     RangeConf<f64> kxc, RangeConf<f64> kyc, RangeConf<f64>& ec,
                     f64 sharpening, f64 cutoff, bool printProgress) {
  std::cout << "Calculating density of states\n";
  const u32 its = ec.n / 10;
  std::vector<f64> densities(ec.n, 0);
  if (fleq(ec.start, ec.end, 1e-16)) {
    autoLimits(D, ec);
  }
  auto del = delta(D, ec, sharpening, cutoff);
  if (printProgress)
    std::cout << "[" << std::flush;
  for (u32 k = 0; k < ec.n; k++) {
    std::vector<f64> pre_densities(kxc.n * kyc.n, 0);
    auto pre_density_view = std::mdspan(pre_densities.data(), kxc.n, kyc.n);
    for (const auto& pair : del[k]) {
#pragma omp parallel for
      for (u32 i = 0; i < kxc.n; i++) {
        const f64 kx = kxc.ith(i) * 2 * M_PI / lat_const;
        for (u32 j = 0; j < kyc.n; j++) {
          const f64 ky = kyc.ith(j) * 2 * M_PI / lat_const;
          const VectorXcd k_vec = UH * planeWave({kx, ky}, points);
          pre_density_view[i, j] += pair.first * std::norm(k_vec(pair.second));
        }
      }
#pragma omp barrier
    }
    for (u32 i = 0; i < kxc.n; i++) {
      for (u32 j = 0; j < kyc.n; j++) {
        densities[k] += pre_density_view[i, j];
      }
    }
    if (printProgress)
      if (k % its == 0)
        std::cout << "█|" << std::flush;
  }
  if (printProgress)
    std::cout << "█]\n";
  return densities;
}

std::vector<f64> disp(const VectorXd& D, const MatrixXcd& UH,
                      const std::vector<Point>& points, f64 lat_const,
                      std::vector<RangeConf<Vector2d>> kc, RangeConf<f64>& ec,
                      f64 sharpening, f64 cutoff, bool printProgress) {
  std::cout << "Calculating dispersion relation\n";
  u32 nsamples = 0;
  for (const auto& r : kc) {
    nsamples += r.n;
  }
  const u32 its = nsamples / 10;
  std::cout << "its: " << its << '\n';
  std::vector<f64> disp(nsamples * ec.n, 0);
  if (fleq(ec.start, ec.end, 1e-16)) {
    autoLimits(D, ec);
  }
  auto del = delta(D, ec, sharpening, cutoff);
  auto disp_view = std::mdspan(disp.data(), ec.n, nsamples);
  if (printProgress)
    std::cout << "[" << std::flush;

  // duration<u64, nanoseconds::period> kvecduration{};
  // duration<u64, nanoseconds::period> dispcalcduration{};
  u32 it = 0;
  u32 count = 0;
  for (const auto& rc : kc) {
    for (u32 i = 0; i < rc.n; i++) {
      const auto k = rc.ith(i);
      // auto start = steady_clock::now();
      const VectorXcd k_vec = UH * planeWave(k, points);
      // auto end = steady_clock::now();
      // kvecduration += end - start;
#pragma omp parallel for
      for (size_t j = 0; j < ec.n; j++) {
        // auto sumstart = steady_clock::now();
        for (const auto& pair : del[j]) {
          disp_view[j, it + i] += pair.first * std::norm(k_vec(pair.second));
        }
        // auto sumend = steady_clock::now();
        // dispcalcduration += sumend - sumstart;
      }
#pragma omp barrier
      ++count;
      if (printProgress)
        if (count % its == 0)
          std::cout << "█|" << std::flush;
    }
    it += rc.n;
    std::cout << count << '\n';
  }
  if (printProgress)
    std::cout << "█]\n";
  // std::cout << "Time to initialize k states: " << kvecduration << '\n';
  // std::cout << "Time to sum disp values: " << dispcalcduration << '\n';
  return disp;
}

MatrixXd pointsToFiniteHamiltonian(const std::vector<Point>& points,
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
  return finite_hamiltonian(points.size(), nb_info, &expCoupling);
}

#define SET_STRUCT_FIELD(c, tbl, key)                                          \
  if (tbl.contains(#key))                                                      \
  c.key = *tbl[#key].value<decltype(c.key)>()

std::optional<SdfConf> tomlToSdfConf(const std::string& tomlPath) {
  toml::table tbl;
  try {
    tbl = toml::parse_file(tomlPath);
  } catch (const std::exception& err) {
    std::cerr << "Parsing file " << tomlPath
              << " failed with exception: " << err.what() << '\n';
    return {};
  }
  SdfConf conf{};
  toml::table& preConf = *tbl["PreConf"].as_table();
  SET_STRUCT_FIELD(conf, preConf, sharpening);
  SET_STRUCT_FIELD(conf, preConf, cutoff);
  SET_STRUCT_FIELD(conf, preConf, searchRadius);
  SET_STRUCT_FIELD(conf, preConf, pointPath);
  SET_STRUCT_FIELD(conf, preConf, H5Filename);
  SET_STRUCT_FIELD(conf, preConf, doFullSDF);
  SET_STRUCT_FIELD(conf, preConf, doDOS);
  SET_STRUCT_FIELD(conf, preConf, doEsection);
  SET_STRUCT_FIELD(conf, preConf, doPath);
  SET_STRUCT_FIELD(conf, preConf, fixed_e);
  if (conf.doPath) {
    try {
      toml::table pathSpec = *tbl["path"].as_table();
      toml::array ranges = *pathSpec["ranges"].as_array();
      for (const auto& range : ranges) {
        toml::table r = *range.as_table();
        conf.kpath.push_back(tblToVecRange(r));
      }
      conf.dispE = tblToRange(*tbl["dispE"].as_table());
    } catch (const std::exception& exc) {
      std::cerr << "Path dispersion requested but no ranges specified: "
                << exc.what() << std::endl;
    }
  }

  if (preConf.contains("saveDiagonalisation"))
    conf.saveDiagonalisation =
        preConf["saveDiagonalisation"].value<std::string>();
  if (preConf.contains("useSavedDiag"))
    conf.useSavedDiag = preConf["useSavedDiag"].value<std::string>();
  if (preConf.contains("saveHamiltonian"))
    conf.saveHamiltonian = preConf["saveHamiltonian"].value<std::string>();
  conf.sectionKx = tblToRange(*tbl["sectionKx"].as_table());
  conf.sectionKy = tblToRange(*tbl["sectionKy"].as_table());
  conf.SDFKx = tblToRange(*tbl["SDFKx"].as_table());
  conf.SDFKy = tblToRange(*tbl["SDFKy"].as_table());
  conf.SDFE = tblToRange(*tbl["SDFE"].as_table());
  return conf;
}
#undef SET_STRUCT_FIELD

int doSDFcalcs(SdfConf& conf) {
  if (!(conf.doPath || conf.doEsection || conf.doDOS || conf.doFullSDF)) {
    std::cout << "No tasks selected.\n";
    return 0;
  }

  std::vector<Point> points;
  if (conf.pointPath == "square") {
    points.resize(70 * 70);
    for (u32 i = 0; i < 70; i++) {
      for (u32 j = 0; j < 70; j++) {
        points[i * 70 + j] = {(double)i, (double)j, i * 70 + j};
      }
    }
  } else
    points = readPoints(conf.pointPath);

  kdt::KDTree kdtree(points);
  double a = avgNNDist(kdtree, points);
  EigenSolution eigsol;
  bool useSavedSucceeded = false;
  if (conf.useSavedDiag.has_value()) {
    const std::string& fname = conf.useSavedDiag.value();
    if (file_exists(fname)) {
      auto result = loadDiag(fname);
      useSavedSucceeded = result.has_value();
      if (useSavedSucceeded)
        eigsol = result.value();
    }
  }
  if (!useSavedSucceeded) {
    MatrixXd H =
        pointsToFiniteHamiltonian(points, kdtree, conf.searchRadius * a);
    if (conf.saveHamiltonian.has_value()) {
      saveEigen(conf.saveHamiltonian.value(), H);
    }
    eigsol = hermitianEigenSolver(H);
  }
  if (conf.saveDiagonalisation.has_value() && !useSavedSucceeded) {
    hid_t file = H5Fcreate(conf.saveDiagonalisation.value().c_str(),
                           H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t sizes[2] = {static_cast<hsize_t>(eigsol.D.size()),
                        static_cast<hsize_t>(2 * eigsol.D.size())};
    writeArray<1>("D", file, eigsol.D.data(), sizes);
    writeArray<2>("U", file, eigsol.U.data(), sizes);
    H5Fclose(file);
  }
  hid_t file = H5Fcreate(conf.H5Filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                         H5P_DEFAULT);
  if (file == H5I_INVALID_HID) {
    std::cerr << "Failed to create file " << conf.H5Filename << std::endl;
    return 1;
  }
  if (conf.doEsection) {
    auto UH = eigsol.U.adjoint();
    auto section =
        Esection(eigsol.D, UH, points, a, conf.sectionKx, conf.sectionKy,
                 conf.fixed_e, conf.sharpening, conf.cutoff);
    hsize_t sizes[2] = {conf.sectionKx.n, conf.sectionKy.n};
    writeArray<2>("section", file, section.data(), sizes);
    double sdfBounds[5] = {conf.sectionKx.start, conf.sectionKx.end,
                           conf.sectionKy.start, conf.sectionKy.end,
                           conf.fixed_e};
    hsize_t boundsize[1] = {5};
    writeArray<1>("section_bounds", file, sdfBounds, boundsize);
  } else if (conf.doFullSDF) {
    auto UH = eigsol.U.adjoint();
    auto sdf = fullSDF(eigsol.D, UH, points, a, conf.SDFKx, conf.SDFKy,
                       conf.SDFE, conf.sharpening, conf.cutoff);
    hsize_t sizes[3] = {conf.SDFKx.n, conf.SDFKy.n, conf.SDFE.n};
    writeArray<3>("sdf", file, sdf.data(), sizes);
    double sdfBounds[6] = {conf.SDFKx.start, conf.SDFKx.end,  conf.SDFKy.start,
                           conf.SDFKy.end,   conf.SDFE.start, conf.SDFE.end};
    hsize_t boundsize[1] = {6};
    writeArray<1>("sdf_bounds", file, sdfBounds, boundsize);
  }
  if (conf.doDOS) {
    auto UH = eigsol.U.adjoint();
    auto dos = DOS(eigsol.D, UH, points, a, conf.SDFKx, conf.SDFKy, conf.SDFE,
                   conf.sharpening, conf.cutoff);
    writeArray<1>("dos", file, dos.data(), &conf.SDFE.n);
    double dosBounds[2] = {conf.SDFE.start, conf.SDFE.end};
    hsize_t boundsize[1] = {2};
    writeArray<1>("dos_bounds", file, dosBounds, boundsize);
  }
  if (conf.doPath) {
    std::cout << "Doing path\n";
    if (conf.dispE.has_value()) {
      auto kc = conf.kpath;
      auto ec = conf.dispE.value();
      auto UH = eigsol.U.adjoint();
      auto dis =
          disp(eigsol.D, UH, points, a, kc, ec, conf.sharpening, conf.cutoff);
      u32 nsamples = 0;
      for (const auto& rc : kc) {
        nsamples += rc.n;
      }
      hsize_t sizes[2] = {nsamples, ec.n};
      writeArray<2>("disp", file, dis.data(), sizes);
      std::vector<f64> dispBounds;
      for (const auto& rc : kc) {
        dispBounds.insert(dispBounds.end(),
                          {rc.start[0], rc.start[1], rc.end[0], rc.end[1]});
      }
      dispBounds.insert(dispBounds.end(), {ec.start, ec.end});
      hsize_t boundsizes[1] = {dispBounds.size()};
      writeArray<1>("disp_bounds", file, dispBounds.data(), boundsizes);
    } else {
      std::cout << "Need to supply a non-zero number of energy samples\n";
    }
  }
  H5Fclose(file);
  return 0;
}
