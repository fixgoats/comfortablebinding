#include "Eigen/Dense"
#include "io.h"
#include <iostream>
#include <toml++/toml.hpp>

struct DynConf {
  std::string pointPath;
  f64 searchRadius;
  f64 t0;
  f64 t1;
};

std::optional<DynConf> tomlToDynConf(const std::string& fname);
int doDynamic(const DynConf& conf);
int doExactBasic(const DynConf& conf);
