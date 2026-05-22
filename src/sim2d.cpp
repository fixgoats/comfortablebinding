#include "SDL3/SDL.h"
#include "SDL3/SDL_vulkan.h"
#include "mathhelpers.h"
#include "metaprogramming.h"
// #include "vkFFT.h"
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
  constexpr RangeConf<f32> x = {.start = -10., .end = 10., .n = 1024};

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

  MetaBuffer cmap = mgr.make_raw_buffer<Vector4d>(256);
  mgr.write_to_buffer(cmap, cm::magma.data(), 4UL * 4 * 256);
  constexpr u32 width = 1920;
  constexpr u32 height = 1080;
  constexpr u32 n_elements = width * height;
  constexpr u32 minm_elements = (n_elements + 15) / 16;
  std::vector<f32> cpu_values(n_elements);
  for (u32 i = n_elements; i > 0; --i) {
    cpu_values[i] = static_cast<f32>(i);
  }
  MetaBuffer values = mgr.vec_to_buffer(cpu_values);
  MetaBuffer minmaxbuf = mgr.make_raw_buffer<f32>(minm_elements);
  Algorithm firstminmax =
      mgr.make_algorithm("build/Shaders/firstminmax.spv", {},
                         {&values, &minmaxbuf}, {bit_cast<f32>(n_elements)});
  Algorithm minmax =
      mgr.make_algorithm("build/Shaders/minmax.spv", {}, {&minmaxbuf},
                         {bit_cast<f32>(minm_elements)});

  VkCommandBuffer cb = mgr.begin_record();
  u32 disp = (n_elements + 31) / 32;
  append_op(cb, firstminmax, disp, 1, 1);
  while (disp > 32) {
    disp = (disp + 31) / 32;
    spdlog::info("Dispatching: {}", disp);
    append_op(cb, minmax, disp, 1, 1);
  }
  vkEndCommandBuffer(cb);
  mgr.execute(cb);

  VmaAllocationCreateInfo alloc_ci{};
  alloc_ci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
  VkImageCreateInfo img_ci{};
  img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  img_ci.imageType = VK_IMAGE_TYPE_2D;
  img_ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  img_ci.extent.width = width;
  img_ci.extent.height = height;
  img_ci.extent.depth = 1;
  img_ci.mipLevels = 1;
  img_ci.arrayLayers = 1;
  img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
  img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  img_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  AllocatedImage img(mgr.allocator, alloc_ci, img_ci);

  VkImageViewCreateInfo view_ci{};
  view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_ci.image = img.img;
  view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  view_ci.subresourceRange.levelCount = 1;
  view_ci.subresourceRange.layerCount = 1;
  VkImageView view{};
  chk_vk(vkCreateImageView(mgr.device, &view_ci, nullptr, &view));

  Algorithm cmap_algo = mgr.make_algorithm("build/Shaders/colormap.spv", {view},
                                           {&cmap, &values, &minmaxbuf},
                                           {bit_cast<f32>(minm_elements)});

  VkCommandBuffer new_cb = mgr.begin_record();
  append_op(new_cb, cmap_algo, (width + 7) / 8, (height + 3) / 4, 1);
  vkEndCommandBuffer(new_cb);
  mgr.execute(new_cb);
  // MetaBuffer gpu_psi = mgr.vec_to_buffer(psi);
  // MetaBuffer gpu_psik = mgr.makeRawBuffer<c32>(x.n * x.n);
  // MetaBuffer gpu_r_prop = mgr.vec_to_buffer(r_prop);
  // MetaBuffer gpu_k_prop = mgr.vec_to_buffer(k_prop);
  // Algorithm linschroedrstep = mgr.make_algorithm(
  //     "Shaders/schroedrstep.spv", {},
  //     std::vector<MetaBuffer*>{&gpu_psi, &gpu_r_prop, &gpu_k_prop});
  // Algorithm linschroedkstep = mgr.make_algorithm(
  //     "Shaders/schroedkstep.spv", {}, {&gpu_psi, &gpu_r_prop, &gpu_k_prop});
  // VkFFTConfiguration conf{};
  // conf.device = pcast<VkDevice>(&mgr.device);
  // conf.physicalDevice = pcast<VkPhysicalDevice>(&mgr.physical_device);
  // conf.FFTdim = 2;
  // conf.size[0] = x.n;
  // conf.size[1] = x.n;
  // conf.queue = pcast<VkQueue>(&mgr.queue);
  // conf.fence = pcast<VkFence>(&mgr.fence);
  // conf.commandPool = pcast<VkCommandPool>(&mgr.command_pool);
  // conf.buffer = pcast<VkBuffer>(&gpu_psi.buffer);
  // conf.bufferSize = &gpu_psi.aInfo.size;
  // conf.normalize = 1;
  // VkFFTApplication app{};
  // initializeVkFFT(&app, conf);
  // auto cb = mgr.begin_record();
  // VkFFTLaunchParams lp{};
  // lp.commandBuffer = pcast<VkCommandBuffer>(&cb);
  // for (u32 i = 0; i < t.n; ++i) {
  //   append_op(cb, linschroedrstep, (x.n * x.n) / WAVE_SIZE, 1, 1);
  //   cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
  //                      vk::PipelineStageFlagBits::eAllCommands, {},
  //                      FULL_MEMORY_BARRIER, nullptr, nullptr);
  //   VkFFTAppend(&app, -1, &lp);
  //   append_op(cb, linschroedkstep, (x.n * x.n) / WAVE_SIZE, 1, 1);
  //   cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
  //                      vk::PipelineStageFlagBits::eAllCommands, {},
  //                      FULL_MEMORY_BARRIER, nullptr, nullptr);
  //   VkFFTAppend(&app, 1, &lp);
  //   append_op(cb, linschroedrstep, (x.n * x.n) / WAVE_SIZE, 1, 1);
  // }
  // cb.end();
  // for (u32 i = 0; i < t.n / MAX_CB_SIZE; ++i) {
  //   mgr.execute(cb);
  // }

  // Renderer renderer(window, mgr, inst.surface);
  // // Algorithm sqnorm_xfer = mgr.make_algorithm(
  // //     "Shaders/xfer.spv", {}, {&gpu_psi, &renderer.value_buffer});
  // SDL_Event event;
  // bool should_quit = false;
  // // auto xfer_cb = mgr.begin_record();
  // // append_op(xfer_cb, sqnorm_xfer, (x.n * x.n) / WAVE_SIZE, 1, 1);
  // // xfer_cb.end();

  // while (!should_quit) {
  //   FrameLimit lim(50);
  //   // mgr.execute(cb);
  //   // mgr.execute(xfer_cb);
  //   renderer.draw_frame();
  //   // check_sdl(SDL_UpdateWindowSurface(window));
  //   while (SDL_PollEvent(&event)) {
  //     if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
  //       should_quit = true;
  //     }
  //   }
  // }

  // // deleteVkFFT(&app);
  SDL_DestroyWindow(window);
  return 0;
}
#undef check_sdl
