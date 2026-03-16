#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "geometry.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include "spdlog/spdlog.h"
#include "vkcore.hpp"
#include <cxxopts.hpp>
#include <gsl/gsl_sf.h>
#include <iostream>
#include <random>

using Eigen::Vector2cd, Eigen::Vector2cf, Eigen::MatrixX2d, Eigen::MatrixX2f,
    Eigen::MatrixX2i, Eigen::MatrixXcf, Eigen::VectorXcf;

struct SpecConsts {
  u32 len;
  u32 wave_size;
  f32 dt;
  f32 alpha;
};

auto hankelDD(const SparseMatrix<c64>& J, f64 p, f64 alpha) {
  return [&, p, alpha](Eigen::VectorXcd psi) {
    return p * psi - c64{1, alpha} * psi.cwiseAbs2().cwiseProduct(psi) +
           J * psi;
  };
}

template <class F, class State>
State rk4step(const State& x, f64 dt, F rhs) {
  spdlog::debug("Function: rk4step.");
  State k1 = rhs(x);
  State k2 = rhs(x + 0.5 * dt * k1);
  State k3 = rhs(x + 0.5 * dt * k2);
  State k4 = rhs(x + dt * k3);
  State ret = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
  spdlog::debug("Exiting rk4step.");
  return ret;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("test program", "bleh");
  options.add_options()("v,verbose", "Verbose output", cxxopts::value<bool>());
  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }

  if (result["v"].count()) {
    spdlog::set_level(spdlog::level::trace);
  }

  Manager mgr(100 * 1024 * 1024);
  vk::PhysicalDeviceSubgroupProperties subprops{};
  vk::PhysicalDeviceProperties2 props{};
  props.pNext = &subprops;
  mgr.physicalDevice.getProperties2(&props);
  const u32 wave_size = subprops.subgroupSize;
  const u32 n = 32 * 32 * 32 + 1;
  std::vector<c32> psi(n, {1, -0.5});
  spdlog::debug("allocating gpu_J");
  MetaBuffer gpu_J = mgr.makeRawBuffer<c32>(1);
  spdlog::debug("allocating gpu_ps");
  MetaBuffer gpu_ps = mgr.makeRawBuffer<f32>(1);
  spdlog::debug("allocating gpu_k1");
  MetaBuffer gpu_k1 = mgr.makeRawBuffer<c32>(1);
  spdlog::debug("allocating gpu_k2");
  MetaBuffer gpu_k2 = mgr.makeRawBuffer<c32>(1);
  spdlog::debug("allocating gpu_k3");
  MetaBuffer gpu_k3 = mgr.makeRawBuffer<c32>(1);
  spdlog::debug("allocating gpu_k4");
  MetaBuffer gpu_k4 = mgr.makeRawBuffer<c32>(1);
  MetaBuffer gpu_row_idx = mgr.makeRawBuffer<u32>(1);
  MetaBuffer gpu_col_idx = mgr.makeRawBuffer<u32>(1);

  MetaBuffer gpu_psi = mgr.vecToBuffer(psi);
  MetaBuffer gpu_psisum = mgr.makeRawBuffer<c32>(n);
  SpecConsts spc{n, wave_size, 0, 0};
  Algorithm alg = mgr.makeAlgorithm<u32>("drivendiss.spv",
                                         {
                                             &gpu_psi,
                                             &gpu_k1,
                                             &gpu_k2,
                                             &gpu_k3,
                                             &gpu_k4,
                                             &gpu_ps,
                                             &gpu_J,
                                             &gpu_row_idx,
                                             &gpu_col_idx,
                                             &gpu_psisum,
                                         },
                                         spc);
  auto cb = mgr.beginRecord();
  u32 baseexp = uintlog2(wave_size);
  u32 stage = 5;
  const u32 reps = uintlog2n(n, baseexp);
  cb.pushConstants(alg.m_PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                   4, &stage);
  appendOp(cb, alg, (n + wave_size - 1) / wave_size, 1, 1);
  stage = 6;
  cb.pushConstants(alg.m_PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                   4, &stage);
  for (u32 i = 0; i < reps; ++i) {
    appendOp(cb, alg, (n + wave_size - 1) / wave_size, 1, 1);
  }
  cb.end();
  mgr.execute(cb);

  c32 psisum;
  mgr.writeFromBuffer(gpu_psisum, &psisum, sizeof(c32));
  std::cout << "psisum: " << psisum << '\n';

  return 0;
}
