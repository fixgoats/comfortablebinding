#include "Eigen/Dense"
#include "vkcore.hpp"
#include <iostream>

using Eigen::Vector2cd, Eigen::Vector2cf;

template <class T>
std::vector<T> range(s64 n) {
  std::vector<T> tmp(n);
  for (s64 i = 0; i < n; ++i) {
    tmp[i] = i;
  }
  return tmp;
}

struct SpecConsts {
  u32 len;
  f32 dt;
  f32 alpha;
};

Vector2cd f(Vector2cd y) {
  return {c64{-0.8 - 1, 1 * std::norm(y[0])} * y[0] + y[1],
          c64{-0.8 - 1, 1 * std::norm(y[1])} * y[1] + y[0]};
}

Vector2cd rk4(Vector2cd y, f64 dt) {
  Vector2cd k1 = f(y);
  Vector2cd k2 = f(y + 0.5 * dt * k1);
  Vector2cd k3 = f(y + 0.5 * dt * k2);
  Vector2cd k4 = f(y + dt * k3);
  return y + (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4);
}

int main(int argc, char* argv[]) {
  const u32 N = 2;
  Manager mgr(100 * 1024 * 1024);
  std::vector<c32> a = {c32{0, 1}, c32{1, 0}};
  std::vector<c32> M{c32{1, 0}, c32{1, 0}};
  std::vector<u32> col_indices{1, 0};
  std::vector<u32> row_indices = {0, 1, 2};
  std::vector<f32> ps = {-0.8, -0.8};
  MetaBuffer gpu_a = mgr.vecToBuffer(a);
  MetaBuffer gpu_M = mgr.vecToBuffer(M);
  MetaBuffer gpu_col_indices = mgr.vecToBuffer(col_indices);
  MetaBuffer gpu_row_indices = mgr.vecToBuffer(row_indices);
  MetaBuffer gpu_ps = mgr.vecToBuffer(ps);
  MetaBuffer gpu_k1 = mgr.makeRawBuffer<c32>(N);
  MetaBuffer gpu_k2 = mgr.makeRawBuffer<c32>(N);
  MetaBuffer gpu_k3 = mgr.makeRawBuffer<c32>(N);
  MetaBuffer gpu_k4 = mgr.makeRawBuffer<c32>(N);

  const SpecConsts spc{N, 0.01, 1.0};
  Algorithm alg = mgr.makeAlgorithm<u32>("drivendiss.spv",
                                         {&gpu_a, &gpu_k1, &gpu_k2, &gpu_k3,
                                          &gpu_k4, &gpu_ps, &gpu_M,
                                          &gpu_row_indices, &gpu_col_indices},
                                         spc);
  vk::CommandBuffer cb = mgr.beginRecord();
  u32 stage = 0;
  cb.pushConstants(alg.m_PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                   4, &stage);
  appendOp(cb, alg, (N + 63) / 64, 1, 1);
  stage = 1;
  cb.pushConstants(alg.m_PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                   4, &stage);
  appendOp(cb, alg, (N + 63) / 64, 1, 1);
  stage = 2;
  cb.pushConstants(alg.m_PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                   4, &stage);
  appendOp(cb, alg, (N + 63) / 64, 1, 1);
  stage = 3;
  cb.pushConstants(alg.m_PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                   4, &stage);
  appendOp(cb, alg, (N + 63) / 64, 1, 1);
  stage = 4;
  cb.pushConstants(alg.m_PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                   4, &stage);
  appendOp(cb, alg, (N + 63) / 64, 1, 1);
  cb.end();
  for (s64 i = 0; i < 10; ++i) {
    mgr.execute(cb);
  }
  mgr.writeFromBuffer(gpu_a, a);
  std::cout << "according to the gpu\n";
  for (s64 i = 0; i < N; i += 1) {
    std::cout << a[i] << '\n';
  }
  Vector2cd v{c64{0, 1}, c64{1, 0}};
  for (s64 i = 0; i < 10; ++i) {
    v = rk4(v, 0.01);
  }
  std::cout << "according to the cpu\n";
  for (s64 i = 0; i < N; i += 1) {
    std::cout << v[i] << '\n';
  }
  std::cout << std::norm(c64{1, 1}) << '\n';
  return 0;
}
