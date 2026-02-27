#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "io.h"
#include "mathhelpers.h"
#include "typedefs.h"
#include <toml++/toml.hpp>

using Eigen::SparseMatrix, Eigen::VectorXcd;

#define SET_STRUCT_FIELD(key, tbl)                                             \
  if (tbl.contains(#key))                                                      \
  key = *tbl[#key].value<decltype(key)>()

struct KuramotoConf {
  std::string outfile;
  f64 K;
  u32 N;
  RangeConf<f64> t;

  KuramotoConf(const toml::table& tbl) {
    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(K, tbl);
    SET_STRUCT_FIELD(N, tbl);
    t = tblToRange(*tbl["t"].as_table());
  }
};

struct BasicConf {
  std::string outfile;
  std::string pointPath;
  std::optional<f64> searchRadius;
  RangeConf<f64> t;

  BasicConf(const toml::table& tbl) {

    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(pointPath, tbl);
    if (tbl.contains("searchRadius")) {
      searchRadius = tbl["searchRadius"].value<f64>().value();
    }
    t = tblToRange(*tbl["t"].as_table());
  }
};

struct BasicDistanceConf {
  std::string outfile;
  f64 alpha;
  f64 p;
  f64 j;
  RangeConf<f64> sep;
  RangeConf<f64> t;

  BasicDistanceConf(const toml::table& tbl) {
    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(alpha, tbl);
    SET_STRUCT_FIELD(p, tbl);
    SET_STRUCT_FIELD(j, tbl);
    t = tblToRange(*tbl["t"].as_table());
    sep = tblToRange(*tbl["sep"].as_table());
  }
};

struct BasicNLinConf {
  std::string outfile;
  std::string pointPath;
  std::optional<f64> searchRadius;
  f64 alpha;
  RangeConf<f64> t;

  BasicNLinConf(const toml::table& tbl) {

    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(pointPath, tbl);
    SET_STRUCT_FIELD(alpha, tbl);
    if (tbl.contains("searchRadius")) {
      searchRadius = tbl["searchRadius"].value<f64>().value();
    }
    t = tblToRange(*tbl["t"].as_table());
  }
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

  TETMConf(const toml::table& tbl) {

    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(pointPath, tbl);
    SET_STRUCT_FIELD(alpha, tbl);
    SET_STRUCT_FIELD(p, tbl);
    SET_STRUCT_FIELD(j, tbl);
    SET_STRUCT_FIELD(rscale, tbl);
    if (tbl.contains("searchRadius")) {
      searchRadius = tbl["searchRadius"].value<f64>().value();
    }
    t = tblToRange(*tbl["t"].as_table());
  }
};

struct HankelScanConf {
  std::string outfile;
  std::string pointPath;
  RangeConf<f64> ps;
  RangeConf<f64> alphas;
  RangeConf<f64> js;
  RangeConf<f64> rscales;
  RangeConf<f64> t;

  HankelScanConf(const toml::table& tbl) {
    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(pointPath, tbl);
    ps = tblToRange(*tbl["ps"].as_table());
    alphas = tblToRange(*tbl["alphas"].as_table());
    js = tblToRange(*tbl["js"].as_table());
    rscales = tblToRange(*tbl["rscales"].as_table());
    t = tblToRange(*tbl["t"].as_table());
  }
#undef SET_STRUCT_FIELD
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
  std::optional<HankelScanConf> hsc;
  std::vector<HankelScanConf> hscs;
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
int doHankelScan(const std::vector<HankelScanConf>& conf);
