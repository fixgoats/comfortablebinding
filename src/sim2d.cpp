#include "SDL3/SDL.h"
#include "SDL3/SDL_vulkan.h"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "vkcore.hpp"
#include <cxxopts.hpp>
#include <iostream>

using Eigen::Vector4d;

namespace {
constexpr f32 v_potential(f32 x, f32 y) {
  return 5 * std::cos(x) * std::sin(y);
}

inline void check_sdl(bool stmt) {
  if (!stmt) {
    SDL_Log("%s", SDL_GetError());
    exit(-1);
  }
}
} // namespace

constexpr u32 WAVE_SIZE = 32;
constexpr u32 MAX_CB_SIZE = 100;

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

#define DEBUG_LOG(...)                                                         \
  spdlog::debug("{}: {}", __func__, std::format(__VA_ARGS__))

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
    exit(0);
  }
  spdlog::set_level(spdlog::level::info);
  if (result["v"].as<bool>()) {
    spdlog::set_level(spdlog::level::debug);
  }
  if (result["vv"].as<bool>()) {
    spdlog::set_level(spdlog::level::trace);
  }

  constexpr u32 nx = 1024;
  constexpr RangeConf<f32> x = {.start = -10., .end = 10., .n = nx};

  DEBUG_LOG("Initializing SDL video and events");
  check_sdl(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS));
  DEBUG_LOG("Running SDL Vulkan loadlibrary");
  check_sdl(SDL_Vulkan_LoadLibrary(nullptr));
  volkInitialize();
  DEBUG_LOG("Creating SDL window");
  auto* window = SDL_CreateWindow("sim2d", 1024, 1024,
                                  SDL_WINDOW_VULKAN | SDL_WINDOW_ALWAYS_ON_TOP);
  u32 sdl_inst_exts_count = 0;
  DEBUG_LOG("Getting SDL's required instance extensions");
  const auto* sdl_inst_exts =
      SDL_Vulkan_GetInstanceExtensions(&sdl_inst_exts_count);
  std::vector<const char*> sdl_inst_exts_vec{
      sdl_inst_exts, sdl_inst_exts + sdl_inst_exts_count};
  AutoInstance inst(sdl_inst_exts_vec, true);
  check_sdl(SDL_Vulkan_CreateSurface(window, *inst, nullptr,
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
  constexpr RangeConf<f32> t = {.start = 0., .end = 0.1, .n = 100};
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
  Renderer renderer(window, mgr, inst.surface, x.n, x.n);
  Algorithm sqnorm_xfer =
      mgr.make_algorithm("build/Shaders/xfer.spv", {},
                         {gpu_psi.address, renderer.value_buf.address});
  SDL_Event event;
  bool should_quit = false;

  VkCommandBuffer xfer_cb = mgr.begin_record();
  append_op(xfer_cb, sqnorm_xfer, (x.n * x.n + 31) / WAVE_SIZE, 1, 1);
  vkEndCommandBuffer(xfer_cb);
  mgr.execute(xfer_cb);

  auto then = std::chrono::high_resolution_clock::now();
  while (!should_quit) {
    FrameLimit lim(17);
    // mgr.execute(cb);
    // mgr.execute(xfer_cb);

    auto now = lim.start;
    u64 now_and_then =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - then)
            .count();
    renderer.vminmax[0] =
        0.4F * square(sinf(0.01F * static_cast<f32>(now_and_then)));
    renderer.vminmax[1] =
        1 - 0.4F * square(cosf(0.01F * static_cast<f32>(now_and_then)));
    memcpy(renderer.vminmax_buf.aInfo.pMappedData, renderer.vminmax.data(), 8);
    renderer.draw_frame();
    // check_sdl(SDL_UpdateWindowSurface(window));
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
        should_quit = true;
      }
    }
  }

  // // deleteVkFFT(&app);
  SDL_DestroyWindow(window);
  return 0;
}
#undef check_sdl
