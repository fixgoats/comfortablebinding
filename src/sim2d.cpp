#include "SDL3/SDL.h"
#include "SDL3/SDL_vulkan.h"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "vkcore.hpp"
#include <cxxopts.hpp>
#include <iostream>
#include <random>

using Eigen::Vector4d;

namespace {
constexpr f32 v_potential(f32 x, f32 y) { return 0.5F * (x * x + y * y); }

inline void check_sdl(bool stmt) {
  if (!stmt) {
    SDL_Log("%s", SDL_GetError());
    exit(-1);
  }
}

inline void check_vkfft(VkFFTResult stmt) {
  if (stmt != VKFFT_SUCCESS) {
    spdlog::error("VkFFT returned error: {}", (s64)stmt);
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
  constexpr RangeConf<f32> x = {.start = -10., .endpoint = 10., .n = nx};

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
  std::random_device dev{};
  std::mt19937 gen(dev());
  std::uniform_real_distribution<f32> dis(-0.0001, 0.0001);

  for (u32 i = 0; i < x.n; ++i) {
    for (u32 j = 0; j < x.n; ++j) {
      // psi[x.n * i + j] = std::exp(
      //     c32{-(square(x.ith(i) - 5.F) + square(x.ith(j) - 5.F)),
      //     -x.ith(i)});
      psi[x.n * i + j] = c32{dis(gen), dis(gen)};
    }
  }
  // std::vector<c64> psik(1024 * 1024);
  constexpr RangeConf<f32> k = {
      .start = -M_PI / x.d(), .endpoint = M_PI / x.d(), .n = nx};
  constexpr RangeConf<f32> t = {.start = 0., .endpoint = 0.1, .n = 100};
  std::vector<c32> k_prop(x.n * x.n);
  for (u32 i = 0; i < k.n; ++i) {
    for (u32 j = 0; j < k.n; ++j) {
      k_prop[i * k.n + j] =
          std::exp(c32{0., -t.d() * (square(k.ith(fftshiftidx(j, x.n))) +
                                     square(k.ith(fftshiftidx(i, x.n))))});
    }
  }
  // std::vector<c32> r_prop(x.n * x.n);
  // for (u32 i = 0; i < k.n; ++i) {
  //   for (u32 j = 0; j < k.n; ++j) {
  //     r_prop[i * x.n + j] =
  //         std::exp(c32{0., -0.5F * t.d() * v_potential(x.ith(i), x.ith(j))});
  //   }
  // }
  std::vector<f32> pump(x.n * x.n);
  for (u32 i = 0; i < x.n; ++i) {
    for (u32 j = 0; j < x.n; ++j) {
      pump[i * x.n + j] = std::exp(-(square(x.ith(j) + x.ith(i))));
    }
  }

  MetaBuffer gpu_psi = mgr.vec_to_buffer(psi);
  MetaBuffer gpu_old_psi = mgr.vec_to_buffer(psi);
  MetaBuffer gpu_pump = mgr.vec_to_buffer(pump);
  MetaBuffer gpu_nr = mgr.make_raw_buffer<f32>(x.n * x.n);
  mgr.default_init_buffer<f32>(gpu_nr, x.n * x.n);
  // MetaBuffer gpu_r_prop = mgr.vec_to_buffer(r_prop);
  MetaBuffer gpu_k_prop = mgr.vec_to_buffer(k_prop);
  Algorithm gpe_first_rstep = mgr.make_algorithm(
      "build/Shaders/gpe_first_rstep.spv", {},
      {gpu_psi.address, gpu_old_psi.address, gpu_nr.address, gpu_pump.address},
      {}, 10);
  // Algorithm schroedrstep =
  //     mgr.make_algorithm("build/Shaders/schroedrstep.spv", {},
  //                        {gpu_psi.address, gpu_r_prop.address});
  Algorithm gpe_last_rstep = mgr.make_algorithm(
      "build/Shaders/gpe_last_rstep.spv", {},
      {gpu_psi.address, gpu_nr.address, gpu_pump.address}, {}, 10);
  Algorithm nr_rk4 = mgr.make_algorithm(
      "build/Shaders/nr_rk4.spv", {},
      {gpu_psi.address, gpu_old_psi.address, gpu_nr.address, gpu_pump.address},
      {}, 10);

  Renderer renderer(window, mgr, inst.surface, x.n, x.n);
  Algorithm sqnorm_xfer =
      mgr.make_algorithm("build/Shaders/xfer.spv", {},
                         {gpu_psi.address, renderer.value_buf.address});
  SDL_Event event;
  bool should_quit = false;

  VkFFTConfiguration conf = mgr.fft_conf<2>({x.n, x.n}, gpu_psi);
  conf.performConvolution = 1;
  conf.kernel = &gpu_k_prop.buffer;
  conf.numberKernels = 1;
  conf.kernelSize = &gpu_k_prop.aInfo.size;
  // conf.kernelConvolution = 1;
  VkFFTApplication app{};
  check_vkfft(initializeVkFFT(&app, conf));
  VkFFTLaunchParams lp{};
  VkCommandBuffer evo_cb = mgr.begin_record();
  lp.commandBuffer = &evo_cb;
  u32 disp_count = (x.n * x.n + 31) / 32;
  VkBufferCopy2 buf_copy{};
  buf_copy.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
  buf_copy.size = gpu_psi.aInfo.size;
  VkCopyBufferInfo2 cpbi{};
  cpbi.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
  cpbi.srcBuffer = gpu_psi.buffer;
  cpbi.dstBuffer = gpu_old_psi.buffer;
  cpbi.regionCount = 1;
  cpbi.pRegions = &buf_copy;
  VkMemoryBarrier2 barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
  barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  VkDependencyInfo dep_info{};
  dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dep_info.memoryBarrierCount = 1;
  dep_info.pMemoryBarriers = &barrier;
  std::array<f32, 6> pars{-1.0, 1.0, 0.5, 1.0, 1.0, 0.01};
  for (u32 i = 0; i < 32; ++i) {
    append_op(evo_cb, gpe_first_rstep, disp_count, 1, 1, pars);
    check_vkfft(VkFFTAppend(&app, -1, &lp));
    append_op(evo_cb, gpe_last_rstep, disp_count, 1, 1, pars);
    append_op(evo_cb, nr_rk4, disp_count, 1, 1);
    vkCmdPipelineBarrier2(evo_cb, &dep_info);
    vkCmdCopyBuffer2(evo_cb, &cpbi);
  }
  append_op(evo_cb, sqnorm_xfer, (x.n * x.n + 31) / WAVE_SIZE, 1, 1);
  vkEndCommandBuffer(evo_cb);

  auto then = std::chrono::high_resolution_clock::now();
  renderer.vminmax[0] = 0;
  renderer.vminmax[1] = 0;
  memcpy(renderer.vminmax_buf.aInfo.pMappedData, renderer.vminmax.data(), 8);
  while (!should_quit) {
    FrameLimit lim(500);

    auto now = lim.start;
    u64 now_and_then =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - then)
            .count();

    renderer.draw_frame();
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
        should_quit = true;
      }
    }
    mgr.execute(evo_cb);
  }

  deleteVkFFT(&app);
  SDL_DestroyWindow(window);
  return 0;
}
#undef check_sdl
