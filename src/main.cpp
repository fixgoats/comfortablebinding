#include "Eigen/Dense"
#include "SDF.h"
#include "approximant.h"
#include "dynamic.h"
#include "geometry.h"
#include "hermEigen.h"
#include "io.h"
#include "kdtree.h"
#include "periodic.h"
#include "vkcore.h"
// #include "lodepng.h"
#include "mathhelpers.h"
#include "typedefs.h"
#include <H5public.h>
#include <cstddef>
#include <cxxopts.hpp>
#include <exception>
#include <iostream>
#include <mdspan>
#include <ostream>
#include <string>
#include <toml++/toml.hpp>

using Eigen::MatrixX2d, Eigen::MatrixXd;

EigenSolution pointsToDiagFormHamiltonian(const std::vector<Point>& points,
                                          const kdt::KDTree<Point>& kdtree,
                                          double rad) {
  auto H = pointsToFiniteHamiltonian(points, kdtree, rad);
  EigenSolution eig = hermitianEigenSolver(H);
  return eig;
}

int main(const int argc, const char* const* argv) {
  cxxopts::Options options("MyProgram", "bleh");
  options.add_options()("p,points", "File name", cxxopts::value<std::string>())(
      "c,config", "TOML configuration", cxxopts::value<std::string>())(
      "t,test", "Test whatever feature I'm working on rn.",
      cxxopts::value<std::string>())("d,dynamic", "Dynamic simulation",
                                     cxxopts::value<std::string>());

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return 1;
  }
  if (result["d"].count()) {
    std::string fname = result["d"].as<std::string>();
    DynConf conf;
    if (auto opt = tomlToDynConf(fname); opt.has_value()) {
      conf = opt.value();
    } else {
      return 1;
    }
    doDynamic(conf);
    doExactBasic(conf);
  }
  if (result["c"].count()) {
    std::string fname = result["c"].as<std::string>();
    SdfConf conf;
    if (auto opt = tomlToSdfConf(fname); opt.has_value()) {
      conf = opt.value();
    } else {
      return 1;
    }
    doSDFcalcs(conf);
  }
  if (result["p"].count()) {
    std::string fname = result["p"].as<std::string>();
    PerConf conf;
    if (auto opt = tomlToPerConf(fname); opt.has_value()) {
      conf = opt.value();
    } else {
      std::cerr << "Couldn't parse file: " << fname << std::endl;
      return 1;
    }
    doPeriodicModel(conf);
  }
  if (result["t"].count()) {
    Manager m(1024);
    std::string fname = result["t"].as<std::string>();
    SdfConf conf;
    if (auto opt = tomlToSdfConf(fname); opt.has_value()) {
      conf = opt.value();
    } else {
      return 1;
    }
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
      const auto UH = eigsol.U.adjoint();
      auto section = GPUEsection(m, eigsol.D, UH, points, a, conf.sectionKx,
                                 conf.sectionKy, conf.fixed_e, conf.sharpening,
                                 conf.cutoff);
      hsize_t sizes[2] = {conf.sectionKx.n, conf.sectionKy.n};
      writeSingleArray<2>("section", file, section.data(), sizes);
      double sdfBounds[5] = {conf.sectionKx.start, conf.sectionKx.end,
                             conf.sectionKy.start, conf.sectionKy.end,
                             conf.fixed_e};
      hsize_t boundsize[1] = {5};
      writeArray<1>("section_bounds", file, sdfBounds, boundsize);
    }
    H5Fclose(file);
    // auto alg = m.makeAlgorithm("Shaders/test.spv", {}, {&gpua, &gpub}, none);
    // auto buffer = m.beginRecord();
    // appendOp(buffer, alg, 1024, 1, 1);
    // buffer.end();
    // m.execute(buffer);
    // m.writeFromBuffer(gpub, b);
    // for (const auto& x : b) {
    //   std::cout << x << '\n';
    // }
    // auto vec = readPoints(result["t"].as<std::string>());
    // constexpr size_t N = 40;
    // std::vector<Point> vec(N * N);
    // for (u32 i = 0; i < N; i++) {
    //   for (u32 j = 0; j < N; j++) {
    //     vec[N * i + j] = Point(i, j, N * i + j);
    //   }
    // }
    // pointsToSDFOnFile(vec, 1.0);
    return 0;
  }
}
