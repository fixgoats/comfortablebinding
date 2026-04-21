#pragma once
#include "betterexc.h"
#include "colormaps.hpp"
#include "hack.hpp"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "typedefs.h"
#include <array>
#include <boost/pfr/core.hpp>
#include <chrono>
#include <cstddef>
#include <format>
#include <fstream>
#include <span>

using std::bit_cast;

constexpr u32 GRID_WIDTH = 512;
constexpr u32 GRID_HEIGHT = 512;

constexpr u32 round_up_x16(u32 n) { return ((n + 15) / 16) * 16; }

struct PositionTextureVertex {
  std::array<f32, 2> pos;
  std::array<f32, 2> uv;

  static vk::VertexInputBindingDescription binding_dscr() {
    return {0, sizeof(PositionTextureVertex), vk::VertexInputRate::eVertex};
  }
  static std::array<vk::VertexInputAttributeDescription, 2> attribute_dscr() {
    return {{{0, 0, vk::Format::eR32G32Sfloat,
              offsetof(PositionTextureVertex, pos)},
             {1, 0, vk::Format::eR32G32Sfloat,
              offsetof(PositionTextureVertex, uv)}}};
  }
};

template <typename T>
std::vector<T> read_file(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw runtime_exc("failed to open file: {}!", filename);
  }

  size_t file_size = static_cast<size_t>(file.tellg());
  std::vector<T> buffer(file_size / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), file_size);
  file.close();
  return buffer;
}

inline std::string tstamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t time = std::chrono::system_clock::to_time_t(now);
  tm local_tm = *localtime(&time);
  return std::format("{}-{}-{}/{}-{}", local_tm.tm_year - 100,
                     local_tm.tm_mon + 1, local_tm.tm_mday, local_tm.tm_hour,
                     local_tm.tm_min);
}

void save_to_file(std::string fname, const char* buf, size_t size);

constexpr u32 MAX_FRAMES_IN_FLIGHT = 2;

/*struct SimConstants {
  u32 nElementsX;
  u32 nElementsY;
  u32 nElementsZ;
  u32 xGroupSize;
  u32 yGroupSize;
  f32 gamma;
  f32 Gamma;
  f32 R;
  f32 EXY;
  f32 dt;
  f32 B0;
  f32 width;
  f32 resgamma;
  f32 Bamp;
  f32 PRatioMax;
  u32 substeps;
  constexpr u32 X() const { return nElementsX / xGroupSize; }
  constexpr u32 Y() const { return nElementsY / yGroupSize; }
  constexpr bool validate() const {
    return (nElementsY % yGroupSize == 0) && (nElementsX % xGroupSize == 0);
  }
  constexpr u32 elementsTotal() const { return nElementsX * nElementsY; }
};*/

// std::ostream& operator<<(std::ostream& os, const SimConstants& obj);
// std::ofstream& operator<<(std::ofstream& os, const SimConstants& obj);

static const vk::MemoryBarrier
    FULL_MEMORY_BARRIER(vk::AccessFlagBits::eMemoryWrite,
                        vk::AccessFlagBits::eMemoryRead);

struct MetaBuffer {
  // A buffer + allocation stuff that you generally need to reference when using
  // vk::Buffers. Also destroys itself automatically.
  VmaAllocator* p_allocator = nullptr;
  vk::Buffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo aInfo;
  MetaBuffer();
  MetaBuffer(VmaAllocator& allocator,
             VmaAllocationCreateInfo& alloc_create_info,
             vk::BufferCreateInfo& bci);
  // To call on default constructed metabuffer
  void allocate(VmaAllocator& allocator,
                VmaAllocationCreateInfo& alloc_create_info,
                vk::BufferCreateInfo& bci);
  ~MetaBuffer();
};

struct AllocatedImage {
  VmaAllocator* p_allocator = nullptr;
  vk::Image img;
  VmaAllocation allocation;
  VmaAllocationInfo aInfo;
  AllocatedImage();
  AllocatedImage(VmaAllocator& allocator,
                 VmaAllocationCreateInfo& alloc_create_info,
                 vk::ImageCreateInfo& ici);
  void allocate(VmaAllocator& allocator,
                VmaAllocationCreateInfo& alloc_create_info,
                vk::ImageCreateInfo& ici);
  ~AllocatedImage();
};

struct SSBO430 {
  u32 n_fields;

  MetaBuffer b;
};

template <class Addr>
constexpr std::span<const f32> spec_span(Addr begin, Addr end) {
  return {bit_cast<f32*>(begin), bit_cast<f32*>(end)};
}

struct Algorithm {
  // Attention, device is used for destroying owned objects, the device will be
  // destroyed by the manager.
  vk::Device m_device;
  // owned
  vk::DescriptorSetLayout m_DSL;
  vk::DescriptorPool m_DescriptorPool;
  vk::DescriptorSet m_DescriptorSet;
  vk::ShaderModule m_ShaderModule;
  vk::PipelineLayout m_PipelineLayout;
  vk::Pipeline m_Pipeline;
  Algorithm() = default;
  Algorithm(vk::Device device, std::span<const u32> spirv, u32 n_imgs,
            u32 n_buffers, u32 n_ubo, std::span<const f32> spec_consts = {},
            size_t n_push_constants = 0);
  void initialize(vk::Device device, std::span<const u32> spirv, u32 n_imgs,
                  u32 n_buffers, u32 n_ubo, std::span<const f32> spec_consts,
                  size_t n_push_constants = 0);
  // Algorithm(vk::Device device, u32 img_views, u32 buffers, u32 n_ubo,
  //           const std::vector<u32>& spirv, const u8* spec_consts = nullptr,
  //           const size_t* sizes = nullptr, size_t n_consts = 0,
  //           const size_t* push_sizes = nullptr, size_t n_push_constants = 0);
  // void initialize(vk::Device device, u32 n_imgs, u32 n_buffers, u32 n_ubo,
  //                 const std::vector<u32>& spirv,
  //                 const u8* spec_consts = nullptr,
  //                 const size_t* sizes = nullptr, size_t n_consts = 0,
  //                 const size_t* push_sizes = nullptr,
  //                 size_t n_push_constants = 0);
  void bind_data(std::span<const vk::ImageView> img_views,
                 std::span<const MetaBuffer* const> buffers,
                 std::span<const MetaBuffer* const> ubos) const;
  ~Algorithm();
};

static const std::vector<std::string> DEVICE_EXTENSIONS = {
    vk::KHRSwapchainExtensionName};
static const std::string APP_NAME{"Vulkan GPE Simulator"};
static const std::string ENGINE_NAME{"argablarg"};

struct AutoInstance {
  vk::Instance instance;
  vk::SurfaceKHR surface = nullptr;

  AutoInstance(std::span<const char*> inst_exts, bool use_layers) {
    vk::ApplicationInfo app_info{APP_NAME.c_str(), 1, ENGINE_NAME.c_str(), 1,
                                 VK_API_VERSION_1_3};
    if (use_layers) {
      spdlog::warn("Vulkan: Validation layers are on");
    }
    const auto layers =
        use_layers ? std::vector<const char*>{"VK_LAYER_KHRONOS_validation"}
                   : std::vector<const char*>{};
    vk::InstanceCreateInfo ici(vk::InstanceCreateFlags(), &app_info, layers,
                               inst_exts);
    try {
      instance = vk::createInstance(ici);
    } catch (vk::SystemError& err) {
      spdlog::error("Manager: {}", err.what());
      exit(-1);
    }
  }
  vk::Instance operator*() const { return instance; }
  [[nodiscard]] constexpr bool has_surface() const {
    return static_cast<bool>(surface);
  }
  ~AutoInstance() { instance.destroy(surface); }
};

struct Manager {
  // vk::Instance instance;
  vk::PhysicalDevice physical_device;
  vk::Device device;
  vk::Queue queue;
  vk::Fence fence;
  VmaAllocator allocator;
  vk::Buffer staging;
  VmaAllocation staging_allocation;
  VmaAllocationInfo staging_info;
  u32 c_qfi = UINT32_MAX;
  u32 g_qfi = UINT32_MAX;
  u32 p_qfi = UINT32_MAX;
  vk::CommandPool command_pool;
  // vk::SurfaceKHR surface;

  Manager(const AutoInstance& instance, size_t staging_size,
          std::span<const char*> extra_device_extensions = {});
  // void finish_setup(size_t staging_size, vk::SurfaceKHR& surface);
  // Manager uses a single staging buffer for efficient copies.
  void copy_buffer(vk::Buffer& src_buffer, vk::Buffer& dst_buffer,
                   u32 buffer_size, u32 src_offset = 0,
                   u32 dst_offset = 0) const;
  void copy_in_batches(vk::Buffer& src_buffer, vk::Buffer& dst_buffer,
                       u32 batch_size, u32 num_batches);
  [[nodiscard]] vk::CommandBuffer copy_op(vk::Buffer src_buffer,
                                          vk::Buffer dst_buffer,
                                          u32 buffer_size, u32 src_offset = 0,
                                          u32 dst_offset = 0) const;

  [[nodiscard]] vk::CommandBuffer
  begin_record(vk::CommandBufferUsageFlagBits bits = {}) const;
  void execute(vk::CommandBuffer& b);
  void execute_no_sync(vk::CommandBuffer& b) const;
  void queue_wait_idle() const;
  // void get_queue_family_indices(vk::SurfaceKHR& surface);
  void write_to_buffer(MetaBuffer& dest, const void* source, size_t size,
                       size_t src_offset = 0, size_t dst_offset = 0);
  template <class T>
  void write_to_buffer(MetaBuffer& buffer, std::vector<T> vec) {
    write_to_buffer(buffer, vec.data(), vec.size() * sizeof(T));
  }
  void write_from_buffer(MetaBuffer& source, void* dest, size_t size);
  template <class T>
  void write_from_buffer(MetaBuffer& buffer, std::vector<T>& v) {
    write_from_buffer(buffer, v.data(), v.size() * sizeof(T));
  }
  template <class T>
  void default_init_buffer(MetaBuffer& buffer, u32 n_elements) {
    T* staging_ptr = bit_cast<T*>(staging_info.pMappedData);
    for (u32 i = 0; i < n_elements; i++) {
      staging_ptr[i] = {};
    }
    copy_buffer(staging, buffer.buffer, n_elements * sizeof(T));
  }
  template <typename T>
  [[nodiscard]] MetaBuffer make_raw_buffer(u32 n_elements) {
    vk::BufferCreateInfo bci{vk::BufferCreateFlags(),
                             round_up_x16(n_elements * sizeof(T)),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc,
                             vk::SharingMode::eExclusive,
                             1,
                             &c_qfi};
    VmaAllocationCreateInfo alloc_create_info{};
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_create_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    alloc_create_info.priority = 1.0F;
    return MetaBuffer{allocator, alloc_create_info, bci};
  }
  template <typename T>
  MetaBuffer make_uniform_object() {
    vk::BufferCreateInfo bci{vk::BufferCreateFlags(),
                             sizeof(T),
                             vk::BufferUsageFlagBits::eUniformBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc,
                             vk::SharingMode::eExclusive,
                             1,
                             &c_qfi};
    VmaAllocationCreateInfo alloc_create_info{};
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_create_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    alloc_create_info.priority = 1.0F;
    return MetaBuffer{allocator, alloc_create_info, bci};
  }

  void free_command_buffer(vk::CommandBuffer& b) const {
    device.freeCommandBuffers(command_pool, 1, &b);
  }

  void free_command_buffers(vk::CommandBuffer* b, u32 n) const {
    device.freeCommandBuffers(command_pool, n, b);
  }

  template <typename T>
  [[nodiscard]] MetaBuffer vec_to_buffer(const std::vector<T>& v) {
    auto buffer = make_raw_buffer<T>(v.size());
    write_to_buffer(buffer, v);
    return buffer;
  }

  // In practice you'll always use 4 byte specialization and push constants, so
  // I'll just assume that here. If you use a non-float constant, just bit_cast
  // it to a float.
  [[nodiscard]] Algorithm make_algorithm(
      std::string spirvname, const std::vector<vk::ImageView>& images,
      const std::vector<MetaBuffer*>& buffers,
      std::span<const f32> spec_consts = {}, size_t n_push_constants = 0) const;

  // [[nodiscard]] Algorithm
  // make_algorithm(std::string spirvname,
  //                const std::vector<vk::ImageView>& images,
  //                std::vector<MetaBuffer*> buffers, std::span<f32>
  //                spec_consts) {
  //   return make_algorithm_raw(spirvname, images, buffers,
  //                             bit_cast<const u8*>(&spec_consts),
  //                             sizes.data(), sizes.size());
  // }
  // template <class PushType, class T>
  // [[nodiscard]] Algorithm make_algorithm(std::string spirvname,
  //                                        std::vector<MetaBuffer*> buffers,
  //                                        const T spec_consts) {
  //   constexpr size_t n_spec_consts = boost::pfr::tuple_size_v<T>;
  //   std::array<size_t, n_spec_consts> sizes;
  //   constexpr_for<0, n_spec_consts, 1>([&sizes](auto i) {
  //     sizes[i] = sizeof(boost::pfr::tuple_element_t<i, T>);
  //   });
  //   constexpr size_t n_push_consts = boost::pfr::tuple_size_v<PushType>;
  //   std::array<size_t, n_push_consts> push_sizes;
  //   constexpr_for<0, n_push_consts, 1>([&push_sizes](auto i) {
  //     push_sizes[i] = sizeof(boost::pfr::tuple_element_t<i, PushType>);
  //   });
  //   return make_algorithm_raw(
  //       spirvname, {}, buffers, bit_cast<const u8*>(&spec_consts),
  //       sizes.data(), sizes.size(), push_sizes.data(), push_sizes.size());
  // }
  ~Manager();
};

struct Renderer {
  // non-owned
  Manager* p_mgr;
  //  owned
  vk::RenderPass render_pass;
  vk::Pipeline graphics_pipeline;
  vk::PipelineLayout graphics_pipeline_layout;
  vk::SwapchainKHR swapchain;
  vk::SurfaceKHR surface;
  std::vector<vk::Framebuffer> swapchain_fbs;
  std::vector<vk::Image> swapchain_imgs;
  vk::Format swapchain_img_fmt;
  vk::Extent2D swapchain_extent;
  std::array<u32, 2> render_queue_indices = {UINT32_MAX, UINT32_MAX};
  vk::Queue graphics_queue;
  vk::Queue present_queue;
  std::vector<vk::ImageView> swapchain_img_views;
  std::vector<vk::Semaphore> present_semaphores;
  std::vector<vk::Semaphore> render_semaphores;
  // std::vector<vk::Fence> image_in_flight_fences;
  std::vector<vk::Fence> in_flight_fences;
  vk::CommandPool command_pool;
  std::vector<vk::CommandBuffer> command_buffers;
  vk::CommandBuffer reduction_buffer;
  MetaBuffer vertex_buffer;
  AllocatedImage colormap_img;
  MetaBuffer colormap;
  vk::ImageView colormap_view;
  vk::Sampler colormap_sampler;
  vk::DescriptorSetLayout descriptor_set_layout;
  vk::DescriptorPool descriptor_pool;
  vk::DescriptorSet descriptor_set;
  vk::SurfaceCapabilitiesKHR capabilities;
  vk::SurfaceFormatKHR surface_format;
  vk::PresentModeKHR present_mode;
  MetaBuffer value_buffer;
  MetaBuffer minmax_buffer;
  Algorithm first_minmax_reduction;
  Algorithm minmax_reduction;
  Algorithm fill_colormap_img;
  u32 n_images;
  bool frame_buffer_resized;
  u32 current_frame = 0;

  Renderer(Manager& manager, vk::SurfaceKHR surf, u32 nx, u32 ny);
  void cleanup_swapchain();
  void recreate_swapchain();
  void draw_frame();
  ~Renderer();
};

std::vector<u32> read_file(const std::string& filename);
vk::PhysicalDevice pick_physical_device(const vk::Instance& instance,
                                        s32 desired_gpu = -1);

template <typename Func>
void one_time_submit(vk::Device device, vk::CommandPool cmd_pool,
                     vk::Queue queue, Func func) {
  vk::CommandBuffer cmd_buffer =
      device
          .allocateCommandBuffers(
              {cmd_pool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  cmd_buffer.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  func(cmd_buffer);
  cmd_buffer.end();
  vk::SubmitInfo submit_info(nullptr, nullptr, cmd_buffer);
  queue.submit(submit_info, nullptr);
  queue.waitIdle();
}

void append_op(vk::CommandBuffer b, const Algorithm& a, u32 x, u32 y, u32 z);
void append_op_no_barrier(vk::CommandBuffer b, const Algorithm& a, u32 x,
                          u32 y = 1, u32 z = 1);
