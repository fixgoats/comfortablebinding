#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "geometry.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include "spdlog/spdlog.h"
#include "vkcore.hpp"
#include <gsl/gsl_sf.h>
#include <iostream>
#include <random>

using Eigen::Vector2cd, Eigen::Vector2cf, Eigen::MatrixX2d, Eigen::MatrixX2f,
    Eigen::MatrixX2i, Eigen::MatrixXcf, Eigen::VectorXcf;

struct SpecConsts {
  u32 len;
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
  {
    Manager mgr(100 * 1024 * 1024);
    MatrixX2d points{{-1, 0}, {1, 0}};
    MatrixX2i couplings{{0, 1}};
    spdlog::debug("allocating gpu_psi");
    MetaBuffer gpu_psi = mgr.makeRawBuffer<c32>(points.rows());
    spdlog::debug("allocating gpu_J");
    MetaBuffer gpu_J = mgr.makeRawBuffer<c32>(couplings.rows() * 2);
    spdlog::debug("allocating gpu_ps");
    MetaBuffer gpu_ps = mgr.makeRawBuffer<f32>(points.rows());
    spdlog::debug("allocating gpu_k1");
    MetaBuffer gpu_k1 = mgr.makeRawBuffer<c32>(points.rows());
    spdlog::debug("allocating gpu_k2");
    MetaBuffer gpu_k2 = mgr.makeRawBuffer<c32>(points.rows());
    spdlog::debug("allocating gpu_k3");
    MetaBuffer gpu_k3 = mgr.makeRawBuffer<c32>(points.rows());
    spdlog::debug("allocating gpu_k4");
    MetaBuffer gpu_k4 = mgr.makeRawBuffer<c32>(points.rows());
    std::vector<u32> col_indices(couplings.rows() * 2);
    std::vector<u32> row_indices(points.rows() + 1);
    {
      const SparseMatrix<c64> bleh =
          SparseC(points, couplings, [](Vector2d d) { return c64{1, 0}; });
      for (u32 i = 0; i < points.rows() + 1; ++i) {
        spdlog::debug("Sparse Matrix row index {}: {}", i,
                      bleh.outerIndexPtr()[i]);
        row_indices[i] = static_cast<u32>(bleh.outerIndexPtr()[i]);
      }
      for (u32 i = 0; i < 2 * couplings.rows(); ++i) {
        spdlog::debug("Sparse Matrix col index {}: {}", i,
                      bleh.innerIndexPtr()[i]);
        col_indices[i] = static_cast<u32>(bleh.innerIndexPtr()[i]);
      }
    }
    spdlog::debug("moving col_indices to gpu");
    MetaBuffer gpu_col_indices = mgr.vecToBuffer(col_indices);
    spdlog::debug("moving row_indices to gpu");
    MetaBuffer gpu_row_indices = mgr.vecToBuffer(row_indices);
    s64 samples = 10;
    spdlog::debug("Taking {} samples", samples);
    std::vector<c32> data(samples * points.rows());
    s64 datasize = data.size();
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<f32> dis(0.0, 2 * M_PI);

    for (s32 i = 0; i < 10; ++i) {

      const f64 scale = 0.1 + i * 0.5;
      const SparseMatrix<c32> J =
          SparseCf(points, couplings, [scale](Vector2d d) {
            return c32{static_cast<f32>(gsl_sf_bessel_J0(scale * d.norm())),
                       static_cast<f32>(gsl_sf_bessel_Y0(scale * d.norm()))};
          });
      mgr.writeToBuffer(gpu_J, J.valuePtr(), 2 * couplings.size());
      const auto sol = Eigen::ComplexEigenSolver<MatrixXcf>(MatrixXcf(J));
      auto min_coeff = std::ranges::min_element(
          sol.eigenvalues(), [](c64 a, c64 b) { return a.imag() < b.imag(); });
      const f64 alpha = 1;
      const SpecConsts spc{static_cast<u32>(points.rows()),
                           static_cast<f32>(0.01), static_cast<f32>(alpha)};
      Algorithm alg = mgr.makeAlgorithm<u32>(
          "drivendiss.spv",
          {&gpu_psi, &gpu_k1, &gpu_k2, &gpu_k3, &gpu_k4, &gpu_ps, &gpu_J,
           &gpu_row_indices, &gpu_col_indices},
          spc);
      const f64 p = 0.1;
      const f32 eff_p = (p - 1) * min_coeff->imag();
      std::vector<f32> ps(points.rows(), eff_p);
      mgr.writeToBuffer(gpu_ps, ps);
      spdlog::debug("Made sparse matrix J.");
      spdlog::debug("Allocating psi.");
      VectorXcf psi(points.rows());
      spdlog::debug("Writing random coordinates to psi.");
      for (u64 o = 0; o < psi.size(); ++o) {
        auto x = dis(gen);
        *(psi.data() + o) = {1e-4f * cos(x), 1e-4f * sin(x)};
      }
      mgr.writeToBuffer(gpu_psi, psi.data(), sizeof(c32) * psi.size());
      u32 super_steps = 10000 / 128;
      u32 end_steps = 10000 % 128;
      vk::CommandBuffer super_cb = mgr.beginRecord();
      vk::CommandBuffer end_cb = mgr.beginRecord();
      for (u32 q = 0; q < 16; ++q) {
        u32 stage = 0;
        super_cb.pushConstants(alg.m_PipelineLayout,
                               vk::ShaderStageFlagBits::eCompute, 0, 4, &stage);
        appendOp(super_cb, alg, (psi.size() + 63) / 64, 1, 1);
        stage = 1;
        super_cb.pushConstants(alg.m_PipelineLayout,
                               vk::ShaderStageFlagBits::eCompute, 0, 4, &stage);
        appendOp(super_cb, alg, (psi.size() + 63) / 64, 1, 1);
        stage = 2;
        super_cb.pushConstants(alg.m_PipelineLayout,
                               vk::ShaderStageFlagBits::eCompute, 0, 4, &stage);
        appendOp(super_cb, alg, (psi.size() + 63) / 64, 1, 1);
        stage = 3;
        super_cb.pushConstants(alg.m_PipelineLayout,
                               vk::ShaderStageFlagBits::eCompute, 0, 4, &stage);
        appendOp(super_cb, alg, (psi.size() + 63) / 64, 1, 1);
        stage = 4;
        super_cb.pushConstants(alg.m_PipelineLayout,
                               vk::ShaderStageFlagBits::eCompute, 0, 4, &stage);
        appendOp(super_cb, alg, (psi.size() + 63) / 64, 1, 1);
      }
      {
        u32 stage = 0;
        end_cb.pushConstants(alg.m_PipelineLayout,
                             vk::ShaderStageFlagBits::eCompute, 0, 4, &stage);
        appendOp(end_cb, alg, (psi.size() + 63) / 64, 1, 1);
        stage = 1;
        end_cb.pushConstants(alg.m_PipelineLayout,
                             vk::ShaderStageFlagBits::eCompute, 0, 4, &stage);
        appendOp(end_cb, alg, (psi.size() + 63) / 64, 1, 1);
        stage = 2;
        end_cb.pushConstants(alg.m_PipelineLayout,
                             vk::ShaderStageFlagBits::eCompute, 0, 4, &stage);
        appendOp(end_cb, alg, (psi.size() + 63) / 64, 1, 1);
        stage = 3;
        end_cb.pushConstants(alg.m_PipelineLayout,
                             vk::ShaderStageFlagBits::eCompute, 0, 4, &stage);
        appendOp(end_cb, alg, (psi.size() + 63) / 64, 1, 1);
        stage = 4;
        end_cb.pushConstants(alg.m_PipelineLayout,
                             vk::ShaderStageFlagBits::eCompute, 0, 4, &stage);
        appendOp(end_cb, alg, (psi.size() + 63) / 64, 1, 1);
      }
      super_cb.end();
      end_cb.end();
      for (u32 o = 0; o < super_steps; ++o) {
        mgr.execute(super_cb);
      }
      for (u32 o = 0; o < end_steps; ++o) {
        mgr.execute(end_cb);
      }
      const size_t byteSize = psi.size() * sizeof(c32);

      mgr.writeFromBuffer(gpu_psi, psi.data(), sizeof(c32) * psi.size());
      size_t idx = i * points.rows();
      spdlog::debug("Copying to {}-th complex number.", idx);
      spdlog::debug("Finished calculating.");
      memcpy(data.data() + idx, psi.data(), byteSize);
    }

    HighFive::File file("test.h5", HighFive::File::Truncate);
    auto set = file.createDataSet<c32>("bleh", HighFive::DataSpace({10, 2}));
    set.write_raw(data.data());
  }
  {
    spdlog::debug("Read pointfile.");
    Eigen::MatrixX2d points{{-1, 0}, {1, 0}};
    spdlog::debug("Read points.");
    Eigen::MatrixX2i couplings{{0, 1}};
    spdlog::debug("Read couplings.");

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(0.0, 2 * M_PI);

    s64 samples = 10;
    std::vector<c64> data(samples * points.rows());
    s64 datasize = data.size();
    spdlog::debug("data has {} elements in total.", datasize);
    s64 psize = 10;
    spdlog::debug("number of ps is {}.", psize);
    VectorXcd init_psi(points.rows());
    for (u64 o = 0; o < init_psi.size(); ++o) {
      auto x = dis(gen);
      *(init_psi.data() + o) = {1e-4 * cos(x), 1e-4 * sin(x)};
    }
    for (s64 n = 0; n < 10; ++n) {
      const f64 scale = 0.1 + n * 0.5;
      const SparseMatrix<c64> J =
          SparseC(points, couplings, [scale](Vector2d d) {
            return c64{gsl_sf_bessel_J0(scale * d.norm()),
                       gsl_sf_bessel_Y0(scale * d.norm())};
          });
      const auto sol = Eigen::ComplexEigenSolver<MatrixXcd>(MatrixXcd(J));
      auto min_coeff = std::ranges::min_element(
          sol.eigenvalues(), [](c64 a, c64 b) { return a.imag() < b.imag(); });
      for (s64 k = 0; k < psize; ++k) {
        const f64 p = 0.1;
        const f64 alpha = 1;
        const f64 eff_p = (p - 1) * min_coeff->imag();
        spdlog::debug("Made sparse matrix J.");
        spdlog::debug("Allocating psi.");
        VectorXcd psi = init_psi;
        spdlog::debug("Writing random coordinates to psi.");
        const size_t byteSize = psi.size() * sizeof(c64);
        auto rhs = hankelDD(J, eff_p, alpha);
        spdlog::debug("Running rk4 for {} steps.", 10000);
        for (u32 i = 1; i < 10000 + 1; ++i) {
          psi = rk4step(psi, 0.01, rhs);
        }
        s64 idx = n * points.rows();
        spdlog::debug("Copying to {}-th complex number.", idx);
        spdlog::debug("Finished calculating.");
        memcpy(data.data() + idx, psi.data(), byteSize);
      }
    }

    HighFive::File file("cputest.h5", HighFive::File::Truncate);
    spdlog::debug("Writing samples to file.");

    auto sampleSet = file.createDataSet<c64>(
        "psis", HighFive::DataSpace({10, static_cast<u64>(points.rows())}));
    sampleSet.write_raw(data.data());
    file.createDataSet("points", points);
    file.createDataSet("couplings", couplings);
    spdlog::debug("Exiting doHankelScan.");
  }
  return 0;
}
