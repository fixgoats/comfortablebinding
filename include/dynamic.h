#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "mathhelpers.h"
#include "typedefs.h"
#include <toml++/toml.hpp>

using Eigen::SparseMatrix, Eigen::VectorXcd;

struct KuramotoConf {
  std::string outfile;
  f64 K;
  u32 N;
  RangeConf<f64> t;
};

struct BasicConf {
  std::string outfile;
  std::string pointPath;
  std::optional<f64> searchRadius;
  RangeConf<f64> t;
};

struct BasicDistanceConf {
  std::string outfile;
  f64 alpha;
  f64 p;
  f64 j;
  RangeConf<f64> sep;
  RangeConf<f64> t;
};

struct BasicNLinConf {
  std::string outfile;
  std::string pointPath;
  std::optional<f64> searchRadius;
  f64 alpha;
  RangeConf<f64> t;
};

struct TETMConf {
  std::string outfile;
  std::string pointPath;
  std::optional<f64> searchRadius;
  f64 p;
  f64 alpha;
  f64 j;
  f64 rscale;
  RangeConf<f64> t;
};

struct DelayConf {
  std::string outfile;
  std::string pointPath;
  std::optional<f64> searchRadius;
  f64 p;
  f64 alpha;
  f64 j;
  f64 v;
  RangeConf<f64> t;
};

struct DynConf {
  std::optional<KuramotoConf> kuramoto;
  std::optional<BasicConf> basic;
  std::optional<BasicDistanceConf> bd;
  std::optional<BasicNLinConf> basicnlin;
  std::optional<TETMConf> tetm;
};

std::optional<DynConf> tomlToDynConf(const std::string& fname);

auto basic(const SparseMatrix<c64>& iH);
int doBasic(const BasicConf& conf);
int doBasicNLin(const BasicNLinConf& conf);
int doExactBasic(const BasicConf& conf);
int doKuramoto(const KuramotoConf& conf);
int doTETM(const TETMConf& conf);
int doDistanceScan(const BasicDistanceConf& conf);
int doNoCoupling(const BasicDistanceConf& conf);
int doNCDD(const BasicDistanceConf& conf);
int doBasicHankelDD(const TETMConf& conf);
