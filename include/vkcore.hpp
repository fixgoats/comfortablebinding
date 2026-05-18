#pragma once
#include "betterexc.h"
#include "colormaps.hpp"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "slang/slang-com-ptr.h"
#include "slang/slang.h"
#include "typedefs.h"
#include <SDL3/SDL.h>
#include <array>
#include <boost/pfr/core.hpp>
#include <chrono>
#include <cstddef>
#include <format>
#include <fstream>
#include <span>
#include <vma/vk_mem_alloc.h>
#include <volk/volk.h>
#include <vulkan/vulkan.h>

#define DEBUG_START spdlog::debug("{}: Start", __func__)
#define DEBUG_END spdlog::debug("{}: End", __func__)
#define DEBUG_LOG(...)                                                         \
  spdlog::debug("{}: {}", __func__, std::format(__VA_ARGS__))

using std::bit_cast;

constexpr u32 GRID_WIDTH = 512;
constexpr u32 GRID_HEIGHT = 512;
constexpr VkFormat SWAPCHAIN_IMAGE_FORMAT = VK_FORMAT_B8G8R8A8_SRGB;

constexpr u32 round_up_x16(u32 n) { return ((n + 15) / 16) * 16; }

struct PositionTextureVertex {
  std::array<f32, 2> pos;
  std::array<f32, 2> uv;

  static VkVertexInputBindingDescription binding_dscr() {
    return {0, sizeof(PositionTextureVertex), VK_VERTEX_INPUT_RATE_VERTEX};
  }
  static std::array<VkVertexInputAttributeDescription, 2> attribute_dscr() {
    return {{{.location = 0,
              .binding = 0,
              .format = VK_FORMAT_R32G32_SFLOAT,
              .offset = offsetof(PositionTextureVertex, pos)},
             {.location = 1,
              .binding = 0,
              .format = VK_FORMAT_R32G32_SFLOAT,
              .offset = offsetof(PositionTextureVertex, uv)}}};
  }
};

namespace {
inline void chk(VkResult result) {
  if (result != VK_SUCCESS) {
    spdlog::error("Vulkan call returned an error: {}",
                  static_cast<s64>(result));
    exit(result);
  }
}
} // namespace

template <typename T>
std::vector<T> read_file(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw runtime_exc("failed to open file: {}!", filename);
  }

  size_t file_size = static_cast<size_t>(file.tellg());
  std::vector<T> buffer(file_size / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(file_size));
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

consteval VkMemoryBarrier2 full_mem_barrier() {
  VkMemoryBarrier2 barrier{};
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
  return barrier;
}
static const VkMemoryBarrier2 FULL_MEMORY_BARRIER = full_mem_barrier();

struct MetaBuffer {
  // A buffer + allocation stuff that you generally need to reference when using
  // VkBuffers. Also destroys itself automatically.
  VmaAllocator* p_allocator = nullptr;
  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo aInfo;
  MetaBuffer();
  MetaBuffer(VmaAllocator& allocator,
             VmaAllocationCreateInfo& alloc_create_info,
             VkBufferCreateInfo& bci);
  // To call on default constructed metabuffer
  void allocate(VmaAllocator& allocator,
                VmaAllocationCreateInfo& alloc_create_info,
                VkBufferCreateInfo& bci);
  ~MetaBuffer();
};

struct AllocatedImage {
  VmaAllocator* p_allocator = nullptr;
  VkImage img;
  VmaAllocation allocation;
  VmaAllocationInfo aInfo;
  AllocatedImage();
  AllocatedImage(VmaAllocator& allocator,
                 VmaAllocationCreateInfo& alloc_create_info,
                 VkImageCreateInfo& ici);
  void allocate(VmaAllocator& allocator,
                VmaAllocationCreateInfo& alloc_create_info,
                VkImageCreateInfo& ici);
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
  VkDevice m_device;
  // owned
  VkDescriptorSetLayout m_DSL;
  VkDescriptorPool m_DescriptorPool;
  VkDescriptorSet m_DescriptorSet;
  VkShaderModule m_ShaderModule;
  VkPipelineLayout m_PipelineLayout;
  VkPipeline m_Pipeline;
  Algorithm() = default;
  Algorithm(VkDevice device, std::span<const u32> spirv, u32 n_imgs,
            u32 n_buffers, u32 n_ubo, std::span<const f32> spec_consts = {},
            size_t n_push_constants = 0);
  void initialize(VkDevice device, std::span<const u32> spirv, u32 n_imgs,
                  u32 n_buffers, u32 n_ubo, std::span<const f32> spec_consts,
                  size_t n_push_constants = 0);
  void bind_data(std::span<const VkImageView> img_views,
                 std::span<const MetaBuffer* const> buffers,
                 std::span<const MetaBuffer* const> ubos) const;
  ~Algorithm();
};

static const std::vector<std::string> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};
static const std::string APP_NAME{"Vulkan GPE Simulator"};
static const std::string ENGINE_NAME{"argablarg"};

struct AutoInstance {
  VkInstance instance;
  VkSurfaceKHR surface = nullptr;

  AutoInstance(std::span<const char*> inst_exts, bool use_layers) {
    DEBUG_START;
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "MyRenderer";
    app_info.apiVersion = VK_API_VERSION_1_3;
    if (use_layers) {
      spdlog::warn("Vulkan: Validation layers are on");
    }
    std::array<const char*, 1> validation_layers{"VK_LAYER_KHRONOS_validation"};
    VkInstanceCreateInfo instance_ci{};
    instance_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_ci.pApplicationInfo = &app_info;
    instance_ci.enabledExtensionCount = inst_exts.size();
    instance_ci.ppEnabledExtensionNames = inst_exts.data();
    instance_ci.enabledLayerCount = 1;
    instance_ci.ppEnabledLayerNames = validation_layers.data();
    chk(vkCreateInstance(&instance_ci, nullptr, &instance));
    volkLoadInstance(instance);
  }
  VkInstance operator*() const { return instance; }
  [[nodiscard]] constexpr bool has_surface() const {
    return static_cast<bool>(surface);
  }
  ~AutoInstance() {
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
  }
};

struct Manager {
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  VkFence fence;
  VmaAllocator allocator;
  VkBuffer staging;
  VmaAllocation staging_allocation;
  VmaAllocationInfo staging_info;
  u32 c_qfi = UINT32_MAX;
  u32 gp_qfi = UINT32_MAX;
  VkCommandPool command_pool;
  // VkSurfaceKHR surface;

  Manager(const AutoInstance& instance, size_t staging_size,
          std::span<const char*> extra_device_extensions = {});
  // void finish_setup(size_t staging_size, VkSurfaceKHR& surface);
  // Manager uses a single staging buffer for efficient copies.
  void copy_buffer(VkBuffer& src_buffer, VkBuffer& dst_buffer, u32 buffer_size,
                   u32 src_offset = 0, u32 dst_offset = 0) const;
  void copy_in_batches(VkBuffer& src_buffer, VkBuffer& dst_buffer,
                       u32 batch_size, u32 num_batches);
  void recreate_staging_buffer(size_t size);

  [[nodiscard]] VkCommandBuffer
  begin_record(VkCommandBufferUsageFlagBits bits = {}) const;
  void execute(VkCommandBuffer b);
  void execute_no_sync(VkCommandBuffer b) const;
  void queue_wait_idle() const;
  // void get_queue_family_indices(VkSurfaceKHR& surface);
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
    VkBufferCreateInfo bci{};
    bci.size = round_up_x16(n_elements * sizeof(T));
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bci.queueFamilyIndexCount = 1;
    bci.pQueueFamilyIndices = &c_qfi;
    VmaAllocationCreateInfo alloc_create_info{};
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_create_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    alloc_create_info.priority = 1.0F;
    return MetaBuffer{allocator, alloc_create_info, bci};
  }
  template <typename T>
  MetaBuffer make_uniform_object() {
    VkBufferCreateInfo bci{};
    bci.size = sizeof(T);
    bci.usage =
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE, bci.queueFamilyIndexCount = 1;
    bci.pQueueFamilyIndices = &c_qfi;
    VmaAllocationCreateInfo alloc_create_info{};
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_create_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    alloc_create_info.priority = 1.0F;
    return MetaBuffer{allocator, alloc_create_info, bci};
  }

  void free_command_buffer(VkCommandBuffer& b) const {
    vkFreeCommandBuffers(device, command_pool, 1, &b);
  }

  void free_command_buffers(VkCommandBuffer* b, u32 n) const {
    vkFreeCommandBuffers(device, command_pool, n, b);
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
  [[nodiscard]] Algorithm
  make_algorithm(std::string spirvname, const std::vector<VkImageView>& images,
                 const std::vector<MetaBuffer*>& buffers,
                 std::span<const f32> spec_consts = {},
                 size_t n_push_constants = 0) const;

  ~Manager();
};

struct Renderer {
  SDL_Window* window;
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  VkSurfaceKHR surface;
  VkSurfaceCapabilitiesKHR surface_caps;
  VkExtent2D swapchain_extent;
  VkSwapchainCreateInfoKHR swapchain_ci{};
  VkSwapchainKHR swapchain;
  std::vector<VkImage> swapchain_imgs;
  std::vector<VkImageView> swapchain_img_views;
  VkCommandPool command_pool;
  VkPipeline pipeline;
  VkPipelineLayout pipeline_layout;
  VkImageCreateInfo depth_img_ci{};
  VkFormat depth_fmt;
  VkImage depth_img;
  VmaAllocator allocator{};
  VmaAllocation depth_allocation{};
  VkImageView depth_view;
  std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> cbs;
  std::array<VkFence, MAX_FRAMES_IN_FLIGHT> fences;
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> img_acquired_semaphores;
  std::vector<VkSemaphore> render_complete_semaphores;
  VmaAllocation v_buffer_allocation{};
  VmaAllocationInfo v_buffer_ai{};
  VkDeviceSize v_buf_size;
  VkDeviceSize i_buf_size;
  VkBuffer v_buffer;
  VkDeviceSize index_count;
  struct Texture {
    VmaAllocation allocation{VK_NULL_HANDLE};
    VkImage image{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
    VkSampler sampler{VK_NULL_HANDLE};
  };
  Texture texture{};
  VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptor_set_layout_tex{VK_NULL_HANDLE};
  VkDescriptorSet descriptor_set_tex{VK_NULL_HANDLE};
  Slang::ComPtr<slang::IGlobalSession> slang_global_session;
  u32 frame_index{};

  void make_depth_img_and_view();

  void recreate_swapchain();

  void init_sync_objects();

  VkDescriptorImageInfo load_tex_img();

  [[nodiscard]] VkShaderModule load_main_shader() const;

  Renderer(SDL_Window* window, Manager& mgr, VkSurfaceKHR surf);

  void draw_frame();
};

// struct Renderer {
//   // non-owned
//   Manager* p_mgr;
//   //  owned
//   VkRenderPass render_pass;
//   VkPipeline graphics_pipeline;
//   VkPipelineLayout graphics_pipeline_layout;
//   VkSwapchainKHR swapchain;
//   VkSurfaceKHR surface;
//   std::vector<VkFramebuffer> swapchain_fbs;
//   std::vector<VkImage> swapchain_imgs;
//   VkFormat swapchain_img_fmt;
//   VkExtent2D swapchain_extent;
//   std::array<u32, 2> render_queue_indices = {UINT32_MAX, UINT32_MAX};
//   VkQueue graphics_queue;
//   VkQueue present_queue;
//   std::vector<VkImageView> swapchain_img_views;
//   std::vector<VkSemaphore> present_semaphores;
//   std::vector<VkSemaphore> render_semaphores;
//   // std::vector<VkFence> image_in_flight_fences;
//   std::vector<VkFence> in_flight_fences;
//   VkCommandPool command_pool;
//   std::vector<VkCommandBuffer> command_buffers;
//   VkCommandBuffer reduction_buffer;
//   MetaBuffer vertex_buffer;
//   AllocatedImage colormap_img;
//   MetaBuffer colormap;
//   VkImageView colormap_view;
//   VkSampler colormap_sampler;
//   VkDescriptorSetLayout descriptor_set_layout;
//   VkDescriptorPool descriptor_pool;
//   VkDescriptorSet descriptor_set;
//   VkSurfaceCapabilitiesKHR capabilities;
//   VkSurfaceFormatKHR surface_format;
//   VkPresentModeKHR present_mode;
//   MetaBuffer value_buffer;
//   MetaBuffer minmax_buffer;
//   Algorithm first_minmax_reduction;
//   Algorithm minmax_reduction;
//   Algorithm fill_colormap_img;
//   u32 n_images;
//   bool frame_buffer_resized;
//   u32 current_frame = 0;
//
//   Renderer(Manager& manager, VkSurfaceKHR surf, u32 nx, u32 ny);
//   void cleanup_swapchain();
//   void recreate_swapchain();
//   void draw_frame();
//   ~Renderer();
// };

std::vector<u32> read_file(const std::string& filename);

inline VkCommandBuffer make_cb(VkDevice device, VkCommandPool cmd_pool) {
  VkCommandBuffer cb{};
  VkCommandBufferAllocateInfo cb_ai{};
  cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cb_ai.commandPool = cmd_pool;
  cb_ai.commandBufferCount = 1;

  vkAllocateCommandBuffers(device, &cb_ai, &cb);
  return cb;
}

template <typename Func>
void one_time_submit(VkDevice device, VkCommandPool cmd_pool, VkQueue queue,
                     Func func) {

  VkCommandBuffer cb = make_cb(device, cmd_pool);
  VkCommandBufferBeginInfo cb_bi;
  cb_bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cb_bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &cb_bi);
  func(cb);
  vkEndCommandBuffer(cb);
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cb;
  vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(device, cmd_pool, 1, &cb);
}

void append_op(VkCommandBuffer b, const Algorithm& a, u32 x, u32 y, u32 z);
void append_op_no_barrier(VkCommandBuffer b, const Algorithm& a, u32 x,
                          u32 y = 1, u32 z = 1);
#undef DEBUG_LOG
#undef DEBUG_END
#undef DEBUG_START
