#include "dynamic.h"
#include "hermEigen.h"
#include "spdlog/spdlog.h"
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
                        cxxopts::value<std::string>())("v,verbose",
                                                       "Verbose output");
  cxxopts::ParseResult result;
  spdlog::set_level(spdlog::level::info);
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }

  if (static_cast<bool>(result["v"].count())) {
    spdlog::set_level(spdlog::level::trace);
  }

  if (static_cast<bool>(result["c"].count())) {
    std::string fname = result["c"].as<std::string>();
    auto confs = toml_to_dyn_conf(fname);
    spdlog::debug("Got {} confs", confs.size());
    for (const auto& conf : confs) {
      spdlog::debug("Running conf's run function");
      conf->run();
    }
  }
}
