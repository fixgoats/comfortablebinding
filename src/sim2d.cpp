#include "SDL3/SDL.h"
#include "SDL3/SDL_vulkan.h"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "vkFFT.h"
#include "vkcore.hpp"
#include <cxxopts.hpp>
#include <iostream>

namespace {
constexpr f32 v_potential(f32 x, f32 y) { return square(x) + square(y); }
} // namespace

constexpr u32 WAVE_SIZE = 32;
constexpr u32 MAX_CB_SIZE = 100;

#define SDLCHECK(cmd)                                                          \
  if (!(cmd)) {                                                                \
    SDL_Log("%s", SDL_GetError());                                             \
    return -1;                                                                 \
  }

struct FrameLimit {
  u32 ms_per_frame;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;

  FrameLimit(u32 ms) : ms_per_frame{ms} {
    start = std::chrono::high_resolution_clock::now();
  }

  ~FrameLimit() {
    end = std::chrono::high_resolution_clock::now();
    if (u32 frame_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        frame_duration < ms_per_frame) {
      SDL_Delay(ms_per_frame - frame_duration);
    }
  }
};

int main(int argc, char* argv[]) {
  cxxopts::Options options("2D simulations", "sim2d");
  options.add_options()("h,help",
                        "Help")("v,verbose", "Verbose logging",
                                cxxopts::value<bool>()->default_value("false"))(
      "vv,very-verbose", "Very verbose logging",
      cxxopts::value<bool>()->default_value("false"));

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
  if (result["v"].as<bool>()) {
    spdlog::set_level(spdlog::level::debug);
  }
  if (result["vv"].as<bool>()) {
    spdlog::set_level(spdlog::level::trace);
  }

  constexpr u32 nx = 1024;
  constexpr RangeConf<f32> x = {.start = -10., .end = 10., .n = 1024};

  SDLCHECK(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS));
  SDLCHECK(SDL_Vulkan_LoadLibrary(nullptr));
  auto* window = SDL_CreateWindow("sim2d", 1024, 1024,
                                  SDL_WINDOW_VULKAN | SDL_WINDOW_ALWAYS_ON_TOP);
  u32 sdl_inst_exts_count = 0;
  const auto* sdl_inst_exts =
      SDL_Vulkan_GetInstanceExtensions(&sdl_inst_exts_count);
  std::vector<const char*> sdl_inst_exts_vec{
      sdl_inst_exts, sdl_inst_exts + sdl_inst_exts_count};
  AutoInstance inst(sdl_inst_exts_vec, true);
  SDLCHECK(SDL_Vulkan_CreateSurface(window, *inst, nullptr,
                                    pcast<VkSurfaceKHR>(&inst.surface)));
  Manager mgr(inst, sizeof(c32) * x.n * x.n);
  std::vector<c32> psi(x.n * x.n);
  for (u32 i = 0; i < x.n; ++i) {
    for (u32 j = 0; j < x.n; ++j) {
      psi[x.n * i + j] = std::exp(
          c32{-(square(x.ith(i) - 5.F) + square(x.ith(j) - 5.F)), -x.ith(i)});
    }
  }
  // std::vector<c64> psik(1024 * 1024);
  constexpr RangeConf<f32> k = {
      .start = -M_PI / x.d(), .end = M_PI / x.d(), .n = nx};
  constexpr RangeConf<f32> t = {.start = 0., .end = 10., .n = 1};
  std::vector<c32> k_prop(x.n * x.n);
  for (u32 i = 0; i < k.n; ++i) {
    for (u32 j = 0; j < k.n; ++j) {
      k_prop[i * k.n + j] =
          std::exp(c32{0., -0.5F * t.d() *
                               (square(k.ith(fftshiftidx(j, x.n))) +
                                square(k.ith(fftshiftidx(i, x.n))))});
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
  Algorithm linschroedrstep = mgr.make_algorithm(
      "Shaders/schroedrstep.spv", {},
      std::vector<MetaBuffer*>{&gpu_psi, &gpu_r_prop, &gpu_k_prop});
  Algorithm linschroedkstep = mgr.make_algorithm(
      "Shaders/schroedkstep.spv", {}, {&gpu_psi, &gpu_r_prop, &gpu_k_prop});
  VkFFTConfiguration conf{};
  conf.device = pcast<VkDevice>(&mgr.device);
  conf.physicalDevice = pcast<VkPhysicalDevice>(&mgr.physical_device);
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
  for (u32 i = 0; i < 100; ++i) {
    append_op(cb, linschroedrstep, (x.n * x.n) / WAVE_SIZE, 1, 1);
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                       vk::PipelineStageFlagBits::eAllCommands, {},
                       FULL_MEMORY_BARRIER, nullptr, nullptr);
    VkFFTAppend(&app, -1, &lp);
    append_op(cb, linschroedkstep, (x.n * x.n) / WAVE_SIZE, 1, 1);
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                       vk::PipelineStageFlagBits::eAllCommands, {},
                       FULL_MEMORY_BARRIER, nullptr, nullptr);
    VkFFTAppend(&app, 1, &lp);
    append_op(cb, linschroedrstep, (x.n * x.n) / WAVE_SIZE, 1, 1);
  }
  cb.end();
  // for (u32 i = 0; i < t.n / MAX_CB_SIZE; ++i) {
  //   mgr.execute(cb);
  // }

  Renderer renderer(mgr, inst.surface, x.n, x.n);
  Algorithm sqnorm_xfer = mgr.make_algorithm(
      "Shaders/xfer.spv", {}, {&gpu_psi, &renderer.value_buffer});
  SDL_Event event;
  bool should_quit = false;
  auto xfer_cb = mgr.begin_record();
  append_op(xfer_cb, sqnorm_xfer, (x.n * x.n) / 32, 1, 1);
  xfer_cb.end();

  while (!should_quit) {
    FrameLimit lim(50);
    mgr.execute(cb);
    mgr.execute(xfer_cb);
    renderer.draw_frame();
    // SDLCHECK(SDL_UpdateWindowSurface(window));
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
        should_quit = true;
      }
    }
  }

  SDL_DestroyWindowSurface(window);
  SDL_DestroyWindow(window);
  return 0;
}
#undef SDLCHECK
