#include "SDL3/SDL.h"
#include "vkFFT.h"
#include "vkcore.hpp"
#include <cxxopts.hpp>
#include <iostream>

constexpr f32 V(f32 x, f32 y) { return x * x + y * y; }

int main(int argc, char* argv[]) {
  cxxopts::Options options("2D simulations", "sim2d");
  options.add_options()("h,help", "Help");

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
  if (result["h"].count()) {
    std::cout << options.help() << std::endl;
  }

  constexpr u32 nX = 1024;
  constexpr RangeConf<f32> x = {0., 100., nX};
  Manager mgr(8 * x.n * x.n);
  std::vector<c32> psi(x.n * x.n);
  for (u32 i = 0; i < x.n; ++i) {
    for (u32 j = 0; j < x.n; ++j) {
      psi[x.n * i + j] =
          std::exp(c32{-(square(x.ith(i)) + square(x.ith(j))), 0});
    }
  }
  // std::vector<c64> psik(1024 * 1024);
  constexpr RangeConf<f32> k = {-M_PI / x.d(), M_PI / x.d(), nX};
  constexpr RangeConf<f32> t = {0., 100., 10000};
  std::vector<c32> kProp(x.n * x.n);
  for (u32 i = 0; i < k.n; ++i) {
    for (u32 j = 0; j < k.n; ++j) {
      kProp[i * k.n + j] =
          std::exp(c32{0., -t.d() * (square(k.ith(j)) + square(k.ith(i)))});
    }
  }
  std::vector<c32> rProp(x.n * x.n);
  for (u32 i = 0; i < k.n; ++i) {
    for (u32 j = 0; j < k.n; ++j) {
      rProp[i * x.n + j] =
          std::exp(c32{0., -0.5f * t.d() * V(x.ith(i), x.ith(j))});
    }
  }
  MetaBuffer gpu_psi = mgr.vecToBuffer(psi);
  // MetaBuffer gpu_psik = mgr.makeRawBuffer<c32>(x.n * x.n);
  MetaBuffer gpu_rProp = mgr.vecToBuffer(rProp);
  MetaBuffer gpu_kProp = mgr.vecToBuffer(kProp);
  Algorithm linschroedrstep = mgr.makeAlgorithmRaw(
      "Shaders/schroedrstep.spv", {}, {&gpu_psi, &gpu_rProp, &gpu_kProp});
  Algorithm linschroedkstep = mgr.makeAlgorithmRaw(
      "Shaders/schroedkstep.spv", {}, {&gpu_psi, &gpu_rProp, &gpu_kProp});
  VkFFTConfiguration conf{};
  conf.FFTdim = 2;
  conf.size[0] = x.n;
  conf.size[1] = x.n;
  conf.queue = pcast<VkQueue>(&mgr.queue);
  conf.fence = pcast<VkFence>(&mgr.fence);
  conf.commandPool = pcast<VkCommandPool>(&mgr.commandPool);
  conf.buffer = pcast<VkBuffer>(&gpu_psi.buffer);
  conf.bufferSize = &gpu_psi.aInfo.size;
  conf.normalize = 1;
  VkFFTApplication app;
  initializeVkFFT(&app, conf);
  auto cb = mgr.beginRecord();
  VkFFTLaunchParams lp{};
  lp.commandBuffer = pcast<VkCommandBuffer>(&cb);
  for (u32 i = 0; i < t.n / 100; ++i) {
    appendOp(cb, linschroedrstep, x.n * x.n / 32, 1, 1);
    VkFFTAppend(&app, -1, &lp);
    appendOp(cb, linschroedkstep, x.n * x.n / 32, 1, 1);
    VkFFTAppend(&app, 1, &lp);
    appendOp(cb, linschroedrstep, x.n * x.n / 32, 1, 1);
  }
  cb.end();
  for (u32 i = 0; i < 100; ++i) {
    mgr.execute(cb);
  }
  auto window = SDL_CreateWindow("sim2d", 1024, 1024,
                                 SDL_WINDOW_VULKAN | SDL_WINDOW_ALWAYS_ON_TOP);
  Renderer a(mgr, x.n, x.n);
  return 0;
}
