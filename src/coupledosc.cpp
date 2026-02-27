#include "dynamic.h"
#include "hermEigen.h"
#include "periodic.h"
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
  cxxopts::Options options("Dynamic Simulations", "bleh");
  options.add_options()("c,config", "TOML configuration",
                        cxxopts::value<std::string>());
  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
  if (result["c"].count()) {
    std::string fname = result["c"].as<std::string>();
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
    if (conf.tetm.has_value()) {
      doBasicHankelDD(conf.tetm.value());
    }
    if (!conf.hscs.empty()) {
      doHankelScan(conf.hscs);
    }
    if (conf.bd.has_value()) {
      doDistanceScan(conf.bd.value());
    }
  }
}
