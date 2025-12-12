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
    if (conf.kuramoto.has_value()) {
      doKuramoto(conf.kuramoto.value());
    }
    if (conf.basic.has_value()) {
      doBasic(conf.basic.value());
    }
    if (conf.basicnlin.has_value()) {
      doBasicNLin(conf.basicnlin.value());
    }
    // doExactBasic(conf);
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
    VectorXd a = VectorXd::Ones(100);

    MatrixXd e = (a * a.transpose()).array() + 1;
    std::cout << "dims: (" << e.cols() << ", " << e.rows() << ")\n";
    return 0;
  }
}
