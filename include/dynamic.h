#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "typedefs.h"
#include <toml++/toml.hpp>

using Eigen::SparseMatrix, Eigen::VectorXcd;

struct DynConf {
  std::string pointPath;
  f64 searchRadius;
  f64 t0;
  f64 t1;
};

std::optional<DynConf> tomlToDynConf(const std::string& fname);

auto basic(const SparseMatrix<c64>& iH);
int doDynamic(const DynConf& conf);
int doExactBasic(const DynConf& conf);
int doKuramoto();
