#include "Eigen/Dense"
#include "SDF.h"
#include "approximant.h"
#include "geometry.h"
#include "hermEigen.h"
#include "io.h"
#include "kdtree.h"
#include "periodic.h"
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
      "c,config", "TOML configuration",
      cxxopts::value<std::string>())("v,verbose", "Debug log");

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return 1;
  }
  if (static_cast<bool>(result["v"].count())) {
    spdlog::set_level(spdlog::level::debug);
  }

  if (static_cast<bool>(result["c"].count())) {

    std::string fname = result["c"].as<std::string>();
    auto confs = toml_to_sdf_conf(fname);
    for (const auto& conf : confs) {
      conf->run();
    }
    // SdfConf conf;
    // if (auto opt = toml_to_sdf_conf(fname); opt.has_value()) {
    //   conf = opt.value();
    // } else {
    //   return 1;
    // }
    // do_sdf_calcs(conf);
  }
  if (static_cast<bool>(result["p"].count())) {
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
}
