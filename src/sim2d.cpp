#include "SDL3/SDL.h"
#include "mathhelpers.h"
#include "vkFFT.h"
#include "vkcore.hpp"
#include <cxxopts.hpp>
#include <iostream>

constexpr f32 v_potential(f32 x, f32 y) { return square(x) + square(y); }

constexpr u32 WAVE_SIZE = 32;
constexpr u32 MAX_CB_SIZE = 100;

#define SDLCHECK(cmd)                                                          \
  if (!(cmd)) {                                                                \
    SDL_Log("%s", SDL_GetError());                                             \
    return -1;                                                                 \
  }

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
  if (static_cast<bool>(result["h"].count())) {
    std::cout << options.help() << std::endl;
  }

  constexpr u32 nx = 1024;
  constexpr RangeConf<f32> x = {.start = 0., .end = 100., .n = nx};
  Manager mgr(sizeof(c32) * x.n * x.n);
  std::vector<c32> psi(x.n * x.n);
  for (u32 i = 0; i < x.n; ++i) {
    for (u32 j = 0; j < x.n; ++j) {
      psi[x.n * i + j] =
          std::exp(c32{-(square(x.ith(i)) + square(x.ith(j))), 0});
    }
  }
  // std::vector<c64> psik(1024 * 1024);
  constexpr RangeConf<f32> k = {
      .start = -M_PI / x.d(), .end = M_PI / x.d(), .n = nx};
  constexpr RangeConf<f32> t = {.start = 0., .end = 100., .n = 10000};
  std::vector<c32> k_prop(x.n * x.n);
  for (u32 i = 0; i < k.n; ++i) {
    for (u32 j = 0; j < k.n; ++j) {
      k_prop[i * k.n + j] =
          std::exp(c32{0., -t.d() * (square(k.ith(j)) + square(k.ith(i)))});
    }
  }
  std::vector<c32> r_prop(x.n * x.n);
  for (u32 i = 0; i < k.n; ++i) {
    for (u32 j = 0; j < k.n; ++j) {
      r_prop[i * x.n + j] =
          std::exp(c32{0., -0.5F * t.d() * v_potential(x.ith(i), x.ith(j))});
    }
  }
  MetaBuffer gpu_psi = mgr.vec_to_buffer(psi);
  // MetaBuffer gpu_psik = mgr.makeRawBuffer<c32>(x.n * x.n);
  MetaBuffer gpu_r_prop = mgr.vec_to_buffer(r_prop);
  MetaBuffer gpu_k_prop = mgr.vec_to_buffer(k_prop);
  Algorithm linschroedrstep = mgr.make_algorithm_raw(
      "Shaders/schroedrstep.spv", {}, {&gpu_psi, &gpu_r_prop, &gpu_k_prop});
  Algorithm linschroedkstep = mgr.make_algorithm_raw(
      "Shaders/schroedkstep.spv", {}, {&gpu_psi, &gpu_r_prop, &gpu_k_prop});
  VkFFTConfiguration conf{};
  conf.FFTdim = 2;
  conf.size[0] = x.n;
  conf.size[1] = x.n;
  conf.queue = pcast<VkQueue>(&mgr.queue);
  conf.fence = pcast<VkFence>(&mgr.fence);
  conf.commandPool = pcast<VkCommandPool>(&mgr.command_pool);
  conf.buffer = pcast<VkBuffer>(&gpu_psi.buffer);
  conf.bufferSize = &gpu_psi.aInfo.size;
  conf.normalize = 1;
  VkFFTApplication app;
  initializeVkFFT(&app, conf);
  auto cb = mgr.begin_record();
  VkFFTLaunchParams lp{};
  lp.commandBuffer = pcast<VkCommandBuffer>(&cb);
  for (u32 i = 0; i < t.n / MAX_CB_SIZE; ++i) {
    append_op(cb, linschroedrstep, x.n * x.n / WAVE_SIZE, 1, 1);
    VkFFTAppend(&app, -1, &lp);
    append_op(cb, linschroedkstep, x.n * x.n / WAVE_SIZE, 1, 1);
    VkFFTAppend(&app, 1, &lp);
    append_op(cb, linschroedrstep, x.n * x.n / WAVE_SIZE, 1, 1);
  }
  cb.end();
  for (u32 i = 0; i < t.n / MAX_CB_SIZE; ++i) {
    mgr.execute(cb);
  }
  // auto* window = SDL_CreateWindow("sim2d", 1024, 1024,
  //                                 SDL_WINDOW_VULKAN |
  //                                 SDL_WINDOW_ALWAYS_ON_TOP);
  // SDL_Event event;
  // bool should_quit = false;
  // SDL_Surface* w_surf = SDL_GetWindowSurface(window);
  // SDL_Surface* img = SDL_LoadBMP("pic.bmp");
  // SDL_Rect win_rect = {0, 0, 800, 800};
  // SDLCHECK(SDL_BlitSurface(img, nullptr, w_surf, nullptr));
  // SDLCHECK(SDL_UpdateWindowSurface(window));
  //
  // while (!should_quit) {
  //   auto start = std::chrono::high_resolution_clock::now();
  //   while (SDL_PollEvent(&event)) {
  //     if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
  //       should_quit = true;
  //     }
  //   }
  //   auto end = std::chrono::high_resolution_clock::now();
  //   if (auto frame_duration =
  //           std::chrono::duration_cast<std::chrono::milliseconds>(end -
  //           start)
  //               .count();
  //       frame_duration < 16) {
  //     SDL_Delay(16 - frame_duration);
  //   }
  // }
  //
  // SDL_DestroyWindowSurface(window);
  // SDL_DestroyWindow(window);
  Renderer a(mgr, x.n, x.n);
  return 0;
}
