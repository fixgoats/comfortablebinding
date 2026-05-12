#include "vkcore.hpp"
// #include "colormaps.h"
#include "mathhelpers.h"
#include "vulkan/vulkan.hpp"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <array>
#include <cmath>
#define VOLK_IMPLEMENTATION
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <volk/volk.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
// #include <ranges>
#include "hack.hpp"
#include <set>

#define DEBUG_START spdlog::debug("{}: Start", __func__)
#define DEBUG_END spdlog::debug("{}: End", __func__)
#define DEBUG_LOG(...)                                                         \
  spdlog::debug("{}: {}", __func__, std::format(__VA_ARGS__))

namespace {
inline void chk(VkResult result) {
  if (result != VK_SUCCESS) {
    std::cerr << "Vulkan call returned an error (" << result << ")\n";
    exit(result);
  }
}

inline bool chk_swapchain(VkResult result) {
  DEBUG_START;
  if (result < VK_SUCCESS) {
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      return false;
    }
    std::cerr << "Vulkan call returned an error (" << result << ")\n";
    exit(result);
  }
  DEBUG_END;
  return true;
}

inline void chk_sdl(bool result) {
  if (!result) {
    std::cerr << "Call returned an error\n";
    SDL_Log("%s", SDL_GetError());
    exit(-1);
  }
}

inline VkDevice make_device(VkPhysicalDevice phys_dev,
                            std::span<const u32> qfis, bool support_graphics) {
  DEBUG_START;
  const float qfpriorities{1.0F};
  // DEBUG_LOG("{}", qfis.size());
  // for (const auto& qfi : qfis) {
  //   DEBUG_LOG("{}", qfi);
  // }
  std::vector<VkDeviceQueueCreateInfo> queue_cis(qfis.size());
  for (u32 i = 0; i < qfis.size(); ++i) {
    queue_cis[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_cis[i].queueFamilyIndex = qfis[i];
    queue_cis[i].queueCount = 1;
    queue_cis[i].pQueuePriorities = &qfpriorities;
  }

  VkPhysicalDeviceVulkan12Features enabled_vk12_features{};
  enabled_vk12_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  enabled_vk12_features.descriptorIndexing = VK_TRUE;
  enabled_vk12_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
  enabled_vk12_features.descriptorBindingVariableDescriptorCount = VK_TRUE;
  enabled_vk12_features.runtimeDescriptorArray = VK_TRUE;
  enabled_vk12_features.bufferDeviceAddress = VK_TRUE;
  VkPhysicalDeviceVulkan13Features enabled_vk13_features{};
  enabled_vk13_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  enabled_vk13_features.pNext = &enabled_vk12_features;
  enabled_vk13_features.synchronization2 = VK_TRUE;
  enabled_vk13_features.dynamicRendering = VK_TRUE;
  VkPhysicalDeviceFeatures enabled_vk10_features{};
  const std::vector<const char*> device_extensions{
      VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  VkDeviceCreateInfo device_ci{};
  device_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_ci.queueCreateInfoCount = queue_cis.size();
  device_ci.pQueueCreateInfos = queue_cis.data();
  device_ci.enabledExtensionCount =
      static_cast<uint32_t>(device_extensions.size());
  device_ci.ppEnabledExtensionNames = device_extensions.data();
  if (support_graphics) {
    device_ci.pNext = &enabled_vk13_features;
    enabled_vk10_features.samplerAnisotropy = VK_TRUE;
  }
  device_ci.pEnabledFeatures = &enabled_vk10_features;
  VkDevice device{VK_NULL_HANDLE};
  DEBUG_LOG("Creating device");
  chk(vkCreateDevice(phys_dev, &device_ci, nullptr, &device));
  DEBUG_LOG("{}", (void*)device);
  DEBUG_END;
  return device;
}

inline std::vector<VkImage> get_swapchain_imgs(VkDevice device,
                                               VkSwapchainKHR swapchain) {

  uint32_t image_count{0};
  chk(vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr));
  std::vector<VkImage> swapchain_imgs(image_count);
  chk(vkGetSwapchainImagesKHR(device, swapchain, &image_count,
                              swapchain_imgs.data()));
  return swapchain_imgs;
}

inline VkFormat pick_depth_format(VkPhysicalDevice physical_device) {

  std::vector<VkFormat> depth_format_list{VK_FORMAT_D32_SFLOAT_S8_UINT,
                                          VK_FORMAT_D24_UNORM_S8_UINT};
  VkFormat depth_format{VK_FORMAT_UNDEFINED};
  for (VkFormat& format : depth_format_list) {
    VkFormatProperties2 format_properties{};
    format_properties.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
    vkGetPhysicalDeviceFormatProperties2(physical_device, format,
                                         &format_properties);
    if (static_cast<bool>(
            format_properties.formatProperties.optimalTilingFeatures &
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
      depth_format = format;
      break;
    }
  }
  if (depth_format == VK_FORMAT_UNDEFINED) {
    std::cerr << "Fatal: unable to pick depth format.\n";
    exit(-1);
  }
  return depth_format;
}

u32 get_compute_queue_family_index(vk::PhysicalDevice phys_dev) {
  const auto queue_family_props = phys_dev.getQueueFamilyProperties();
  const auto index = std::ranges::find_if(
      queue_family_props.cbegin(), queue_family_props.cend(),
      [](vk::QueueFamilyProperties qfp) {
        return static_cast<bool>(qfp.queueFlags & vk::QueueFlagBits::eCompute);
      });
  return std::distance(queue_family_props.begin(), index);
}

std::array<u32, 2>
get_graphics_present_queue_family_indices(vk::PhysicalDevice phys_dev,
                                          vk::SurfaceKHR surface) {
  auto queue_family_props = phys_dev.getQueueFamilyProperties();
  u32 g_qfi = UINT32_MAX;
  u32 p_qfi = UINT32_MAX;
  for (u32 i = 0; i < queue_family_props.size(); i++) {
    if (queue_family_props[i].queueFlags & vk::QueueFlagBits::eGraphics &&
        static_cast<bool>(phys_dev.getSurfaceSupportKHR(i, surface))) {
      g_qfi = i;
      p_qfi = i;
      break;
    }
  }
  if (g_qfi == UINT32_MAX) {
    throw runtime_exc("Fatal: Unable to find graphics queue family index.");
  }
  if (p_qfi == UINT32_MAX) {
    throw runtime_exc("Fatal: Unable to find present queue family index.");
  }
  return {g_qfi, p_qfi};
}

void record_drawing_commands(
    vk::Framebuffer framebuffer, vk::RenderPass render_pass,
    vk::Extent2D swap_extent, vk::Pipeline graphics_pipeline,
    vk::PipelineLayout gpl, vk::DescriptorSet descriptor_set,
    vk::Buffer vertex_buffer, vk::CommandBuffer command_buffer) {
  vk::ClearValue clear{{1.0F, 1.0F, 1.0F, 1.0F}};
  vk::RenderPassBeginInfo render_pass_info(
      render_pass, framebuffer, vk::Rect2D(vk::Offset2D(0, 0), swap_extent),
      clear);
  vk::Viewport viewport(0.0F, 0.0F, (f32)swap_extent.width,
                        (f32)swap_extent.height, 0.0F, 1.0F);
  vk::Rect2D scissor(vk::Offset2D(0, 0), swap_extent);
  command_buffer.setViewport(0, viewport);
  command_buffer.setScissor(0, scissor);
  command_buffer.beginRenderPass(render_pass_info,
                                 vk::SubpassContents::eInline);
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                              graphics_pipeline);
  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, gpl, 0,
                                    descriptor_set, nullptr);
  command_buffer.bindVertexBuffers(0, vertex_buffer, {0});
  command_buffer.draw(6, 1, 0, 0);
  command_buffer.endRenderPass();
}

std::set<std::string> get_supported_extensions() {
  uint32_t count = 0;
  vk::Result result =
      vk::enumerateInstanceExtensionProperties(nullptr, &count, nullptr);
  if (result != vk::Result::eSuccess) {
    throw runtime_exc("Couldn't enumerate instance extension properties.\n");
  }

  std::vector<vk::ExtensionProperties> extension_properties(count);

  // Get the extensions
  result = vk::enumerateInstanceExtensionProperties(
      nullptr, &count, extension_properties.data());
  if (result != vk::Result::eSuccess) {
    throw runtime_exc(
        "Couldn't write instance extension properties to buffer.\n");
  }

  std::set<std::string> extensions;
  for (auto& extension : extension_properties) {
    extensions.insert(extension.extensionName);
  }

  return extensions;
}

void copy_buffer_now(vk::Device device, vk::CommandPool command_pool,
                     vk::Queue queue, vk::Fence fence, vk::Buffer& src_buffer,
                     vk::Buffer& dst_buffer, u32 buffer_size, u32 src_offset,
                     u32 dst_offset) {
  auto cmd_buffer = device
                        .allocateCommandBuffers(
                            {command_pool, vk::CommandBufferLevel::ePrimary, 1})
                        .front();
  vk::CommandBufferBeginInfo cbbi(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  cmd_buffer.begin(cbbi);
  cmd_buffer.copyBuffer(src_buffer, dst_buffer,
                        vk::BufferCopy(src_offset, dst_offset, buffer_size));
  cmd_buffer.end();
  vk::SubmitInfo submit_info(nullptr, nullptr, cmd_buffer);
  queue.submit(submit_info, fence);
  auto result = device.waitForFences(fence, vk::True, -1);
  result = device.resetFences(1, &fence);
  device.freeCommandBuffers(command_pool, cmd_buffer);
}

vk::SwapchainKHR create_swapchain(vk::Device device, vk::SurfaceKHR surface,
                                  vk::SurfaceCapabilitiesKHR capabilities,
                                  vk::SurfaceFormatKHR surface_format,
                                  vk::PresentModeKHR present_mode,
                                  vk::Extent2D swapchain_extent, u32 n_images,
                                  std::array<u32, 2> qfis,
                                  vk::SwapchainKHR old_swapchain = nullptr) {
  vk::SurfaceTransformFlagBitsKHR pre_transform =
      (capabilities.supportedTransforms &
       vk::SurfaceTransformFlagBitsKHR::eIdentity)
          ? vk::SurfaceTransformFlagBitsKHR::eIdentity
          : capabilities.currentTransform;

  vk::CompositeAlphaFlagBitsKHR composite_alpha =
      (capabilities.supportedCompositeAlpha &
       vk::CompositeAlphaFlagBitsKHR::ePreMultiplied)
          ? vk::CompositeAlphaFlagBitsKHR::ePreMultiplied
      : (capabilities.supportedCompositeAlpha &
         vk::CompositeAlphaFlagBitsKHR::ePostMultiplied)
          ? vk::CompositeAlphaFlagBitsKHR::ePostMultiplied
      : (capabilities.supportedCompositeAlpha &
         vk::CompositeAlphaFlagBitsKHR::eInherit)
          ? vk::CompositeAlphaFlagBitsKHR::eInherit
          : vk::CompositeAlphaFlagBitsKHR::eOpaque;
  vk::SwapchainCreateInfoKHR create_info(
      {}, surface, n_images, surface_format.format, surface_format.colorSpace,
      swapchain_extent, 1, vk::ImageUsageFlagBits::eColorAttachment,
      vk::SharingMode::eExclusive, {}, pre_transform, composite_alpha,
      present_mode, vk::True, old_swapchain);
  create_info.queueFamilyIndexCount = 1;
  create_info.pQueueFamilyIndices = qfis.data();
  if (qfis[0] != qfis[1]) {
    // If the graphics and present queues are from different queue families,
    // we either have to explicitly transfer ownership of images between the
    // queues, or we have to create the swapchain with imageSharingMode as
    // VK_SHARING_MODE_CONCURRENT
    create_info.imageSharingMode = vk::SharingMode::eConcurrent;
    create_info.queueFamilyIndexCount = 2;
  }
  return device.createSwapchainKHR(create_info);
}

std::vector<vk::ImageView>
create_image_views(vk::Device device, vk::SurfaceFormatKHR surface_format,
                   std::vector<vk::Image> images) {
  std::vector<vk::ImageView> swapchain_image_views;
  swapchain_image_views.reserve(images.size());
  vk::ImageViewCreateInfo img_view_create_info(
      {}, {}, vk::ImageViewType::e2D, surface_format.format, {},
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  for (auto img : images) {
    img_view_create_info.image = img;
    swapchain_image_views.push_back(
        device.createImageView(img_view_create_info));
  }
  return swapchain_image_views;
}

inline VkBool32 supports_required_features(VkPhysicalDevice phys_dev,
                                           bool graphics) {
  VkPhysicalDeviceFeatures2 supported_features;
  supported_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  VkPhysicalDeviceVulkan12Features supported_12_features;
  supported_12_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  VkPhysicalDeviceVulkan13Features supported_13_features{};
  supported_13_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

  supported_features.pNext = &supported_12_features;
  supported_12_features.pNext = &supported_13_features;
  vkGetPhysicalDeviceFeatures2(phys_dev, &supported_features);
  // phys_dev.getFeatures2(&supported_features);
  DEBUG_LOG("samplerAnisotropy supported: {}",
            supported_features.features.samplerAnisotropy);
  DEBUG_LOG("descriptorIndexing supported: {}",
            supported_12_features.descriptorIndexing);
  DEBUG_LOG("shaderSampledImageArrayNonUniformIndexing supported: {}",
            supported_12_features.shaderSampledImageArrayNonUniformIndexing);
  DEBUG_LOG("descriptorBindingVariableDescriptorCount supported: {}",
            supported_12_features.descriptorBindingVariableDescriptorCount);
  DEBUG_LOG("runtimeDescriptorArray supported: {}",
            supported_12_features.runtimeDescriptorArray);
  DEBUG_LOG("bufferDeviceAddress supported: {}",
            supported_12_features.bufferDeviceAddress);
  DEBUG_LOG("maintenance4 supported: {}", supported_13_features.maintenance4);
  DEBUG_LOG("dynamicRendering supported: {}",
            supported_13_features.dynamicRendering);

  if (graphics) {
    return (supported_features.features.samplerAnisotropy &
            supported_12_features.descriptorIndexing &
            supported_12_features.shaderSampledImageArrayNonUniformIndexing &
            supported_12_features.descriptorBindingVariableDescriptorCount &
            supported_12_features.runtimeDescriptorArray &
            supported_12_features.bufferDeviceAddress &
            supported_13_features.maintenance4 &
            supported_13_features.synchronization2 &
            supported_13_features.dynamicRendering);
  }
  return (supported_12_features.descriptorIndexing &
          supported_12_features.descriptorBindingVariableDescriptorCount &
          supported_12_features.runtimeDescriptorArray &
          supported_12_features.bufferDeviceAddress &
          supported_13_features.maintenance4 &
          supported_13_features.synchronization2);
}

std::vector<VkPhysicalDevice> enumerate_physical_devices(VkInstance instance) {
  uint32_t device_count{0};
  chk(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));
  std::vector<VkPhysicalDevice> devices(device_count);
  chk(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()));
  return devices;
}

inline void warn_on_integrated(VkPhysicalDevice dev) {
  VkPhysicalDeviceProperties phys_device_props;
  vkGetPhysicalDeviceProperties(dev, &phys_device_props);
  if (phys_device_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU or
      phys_device_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
    spdlog::warn("pick_physical_device: Only integrated GPU or CPU detected, "
                 "you may not see much benefit from hardware 'acceleration.'");
  }
}

inline void tally_discrete_and_vram(std::span<VkPhysicalDevice> phys_devices,
                                    std::vector<u32>& discrete,
                                    std::span<u64> vram) {
  u32 n_devices = phys_devices.size();
  for (uint32_t i = 0; i < n_devices; i++) {
    VkPhysicalDeviceProperties phys_device_props;
    vkGetPhysicalDeviceProperties(phys_devices[i], &phys_device_props);
    if (phys_device_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      discrete.push_back(i);
    }

    // Gather reported VRAM sizes as an index to rank GPUs by.
    VkPhysicalDeviceMemoryProperties phys_dev_mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_devices[i], &phys_dev_mem_props);
    for (const auto& heap : phys_dev_mem_props.memoryHeaps) {
      if (static_cast<bool>(heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
        vram[i] = heap.size;
      }
    }
    std::cout << "Found device: " << phys_device_props.deviceName
              << " with vram: " << vram[i] << '\n';
  }
}

VkPhysicalDevice pick_physical_device(VkInstance instance, bool graphics = true,
                                      const s32 desired_gpu = -1) {
  // check if there are GPUs that support Vulkan and "intelligently" select
  // one. Prioritises discrete GPUs, and after that VRAM size.

  std::vector<VkPhysicalDevice> phys_devices =
      enumerate_physical_devices(instance);
  uint32_t n_devices = phys_devices.size();

  // shortcut if there's only one device available.
  if (n_devices == 1) {
    warn_on_integrated(phys_devices[0]);
    if (static_cast<bool>(
            supports_required_features(phys_devices[0], graphics))) {
      DEBUG_END;
      return phys_devices[0];
    }
    std::cerr
        << "Fatal: Only available physical device doesn't support required "
           "features\n";
    exit(-1);
  }

  // Try to select desired GPU if specified.
  if (desired_gpu > -1) {
    if (desired_gpu < static_cast<int32_t>(n_devices)) {
      if (static_cast<bool>(supports_required_features(
              phys_devices[desired_gpu], graphics))) {
        DEBUG_END;
        return phys_devices[desired_gpu];
      }
      std::cerr << "Warning: Selected physical device doesn't support required "
                   "features\n";
    }
    // spdlog::warn("pick_physical_device: Selected device is not available.");
  }

  std::vector<u32> discrete; // the indices of the available discrete gpus
  std::vector<u64> vram(n_devices);
  tally_discrete_and_vram(phys_devices, discrete, vram);
  // only consider discrete gpus if available:
  if (discrete.size() > 0) {
    if (discrete.size() == 1) {
      if (static_cast<bool>(supports_required_features(
              phys_devices[discrete[0]], graphics))) {
        DEBUG_END;
        return phys_devices[discrete[0]];
      }
    }
    const auto max_vram_ptr = std::ranges::max_element(vram);
    VkPhysicalDevice selected_phys_dev =
        phys_devices[std::ranges::distance(vram.begin(), max_vram_ptr)];
    if (static_cast<bool>(
            supports_required_features(selected_phys_dev, graphics))) {
      DEBUG_END;
      return selected_phys_dev;
    }
  }

  // Otherwise pick integrated with the highest VRAM
  const auto max_vram_ptr = std::ranges::max_element(vram);
  VkPhysicalDevice selected_phys_dev =
      phys_devices[std::ranges::distance(vram.begin(), max_vram_ptr)];
  if (static_cast<bool>(
          supports_required_features(selected_phys_dev, graphics))) {
    DEBUG_END;
    return selected_phys_dev;
  }
  std::cerr << "Fatal: No suitable physical device found\n";
  exit(-1);
}

} // namespace

MetaBuffer::MetaBuffer() : allocation{}, aInfo{} {}

MetaBuffer::MetaBuffer(VmaAllocator& allocator,
                       VmaAllocationCreateInfo& alloc_create_info,
                       vk::BufferCreateInfo& bci)
    : p_allocator{&allocator}, allocation{} {
  aInfo = VmaAllocationInfo{};
  vmaCreateBuffer(allocator, pcast<VkBufferCreateInfo>(&bci),
                  &alloc_create_info, pcast<VkBuffer>(&buffer), &allocation,
                  &aInfo);
}

void MetaBuffer::allocate(VmaAllocator& allocator,
                          VmaAllocationCreateInfo& alloc_create_info,
                          vk::BufferCreateInfo& bci) {
  p_allocator = &allocator;
  vmaCreateBuffer(allocator, pcast<VkBufferCreateInfo>(&bci),
                  &alloc_create_info, pcast<VkBuffer>(&buffer), &allocation,
                  &aInfo);
}

MetaBuffer::~MetaBuffer() {
  vmaDestroyBuffer(*p_allocator, buffer, allocation);
}

AllocatedImage::AllocatedImage() : allocation{} {
  img = vk::Image{};
  aInfo = VmaAllocationInfo{};
}

AllocatedImage::AllocatedImage(VmaAllocator& allocator,
                               VmaAllocationCreateInfo& alloc_create_info,
                               vk::ImageCreateInfo& ici)
    : p_allocator{&allocator}, allocation{} {
  img = vk::Image{};
  aInfo = VmaAllocationInfo{};
  vmaCreateImage(allocator, pcast<VkImageCreateInfo>(&ici), &alloc_create_info,
                 pcast<VkImage>(&img), &allocation, &aInfo);
}

void AllocatedImage::allocate(VmaAllocator& allocator,
                              VmaAllocationCreateInfo& alloc_create_info,
                              vk::ImageCreateInfo& ici) {
  p_allocator = &allocator;
  vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&ici),
                 &alloc_create_info, reinterpret_cast<VkImage*>(&img),
                 &allocation, &aInfo);
}

AllocatedImage::~AllocatedImage() {
  vmaDestroyImage(*p_allocator, img, allocation);
}

Algorithm::Algorithm(vk::Device device, std::span<const u32> spirv, u32 n_imgs,
                     u32 n_buffers, u32 n_ubo, std::span<const f32> spec_consts,
                     size_t n_push_constants) {
  initialize(device, spirv, n_imgs, n_buffers, n_ubo, spec_consts,
             n_push_constants);
}

Algorithm::~Algorithm() {
  m_device.destroyDescriptorSetLayout(m_DSL);
  m_device.destroyDescriptorPool(m_DescriptorPool);
  m_device.destroyShaderModule(m_ShaderModule);
  m_device.destroyPipeline(m_Pipeline);
  m_device.destroyPipelineLayout(m_PipelineLayout);
}

// vk::PhysicalDevice pick_physical_device(const vk::Instance& instance,
//                                         const s32 desired_gpu) {
//   // check if there are GPUs that support Vulkan and "intelligently" select
//   // one. Prioritises discrete GPUs, and after that VRAM size.
//   std::vector<vk::PhysicalDevice> p_devices =
//       instance.enumeratePhysicalDevices();
//   uint32_t n_devices = p_devices.size();
//
//   // shortcut if there's only one device available.
//   if (n_devices == 1) {
//     if (p_devices[0].getProperties().deviceType ==
//             vk::PhysicalDeviceType::eIntegratedGpu or
//         p_devices[0].getProperties().deviceType ==
//             vk::PhysicalDeviceType::eCpu) {
//       spdlog::warn(
//           "pick_physical_device: Only integrated GPU or CPU detected, "
//           "you may not see much benefit from hardware 'acceleration.'");
//     }
//     return p_devices[0];
//   }
//
//   // Try to select desired GPU if specified.
//   if (desired_gpu > -1) {
//     if (desired_gpu < static_cast<int32_t>(n_devices)) {
//       return p_devices[desired_gpu];
//     }
//     spdlog::warn("pick_physical_device: Selected device is not available.");
//   }
//
//   std::vector<uint32_t> discrete; // the indices of the available discrete
//   gpus std::vector<uint64_t> vram(n_devices); for (uint32_t i = 0; i <
//   n_devices; i++) {
//     if (p_devices[i].getProperties().deviceType ==
//         vk::PhysicalDeviceType::eDiscreteGpu) {
//       discrete.push_back(i);
//     }
//
//     // Gather reported VRAM sizes as an index to rank GPUs by.
//     auto heaps = p_devices[i].getMemoryProperties().memoryHeaps;
//     for (const auto& heap : heaps) {
//       if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
//         vram[i] = heap.size;
//       }
//     }
//   }
//
//   // only consider discrete gpus if available:
//   if (discrete.size() > 0) {
//     if (discrete.size() == 1) {
//       return p_devices[discrete[0]];
//     }
//     const auto max_vram_ptr = std::ranges::max_element(vram);
//     return p_devices[std::ranges::distance(vram.begin(), max_vram_ptr)];
//   }
//
//   // Otherwise pick integrated with the highest VRAM
//   const auto max_vram_ptr = std::ranges::max_element(vram);
//   return p_devices[std::ranges::distance(vram.begin(), max_vram_ptr)];
// }

Manager::Manager(const AutoInstance& instance, size_t staging_size,
                 std::span<const char*> extra_device_extensions) {
  // Validation layers are extremely helpful, we'll only turn them off if we
  // want absolute maximum performance.

  std::vector<const char*> device_extensions{"VK_KHR_maintenance4"};
  for (const auto& ext : extra_device_extensions) {
    device_extensions.push_back(ext);
  }
  physical_device = pick_physical_device(*instance, instance.has_surface());
  c_qfi = get_compute_queue_family_index(physical_device);
  std::set<u32> qfis;
  qfis.insert(c_qfi);
  spdlog::debug("Manager: Does instance have surface?: {}",
                instance.has_surface());
  if (instance.has_surface()) {
    const auto gp_qfis = get_graphics_present_queue_family_indices(
        physical_device, instance.surface);
    g_qfi = gp_qfis[0];
    p_qfi = gp_qfis[1];
    qfis.insert({g_qfi, p_qfi});
    device_extensions.push_back("VK_KHR_swapchain");
  }

  spdlog::debug("Manager: qfis size: {}", qfis.size());
  spdlog::debug("Manager: queue family indices:");
  for (const auto& idx : qfis) {
    spdlog::debug("Queue family index: {}", idx);
  }

  float queue_priority = 1.0F;

  std::vector<vk::DeviceQueueCreateInfo> dqci;
  dqci.reserve(qfis.size());
  for (const auto idx : qfis) {
    dqci.push_back(vk::DeviceQueueCreateInfo({}, idx, 1, &queue_priority));
  }
  vk::PhysicalDeviceVulkan13Features phys_dev_features13;
  phys_dev_features13.maintenance4 = vk::True;
  vk::PhysicalDeviceFeatures phys_dev_features;
  phys_dev_features.shaderFloat64 = vk::True;
  phys_dev_features.shaderInt64 = vk::True;
  vk::DeviceCreateInfo dci(vk::DeviceCreateFlags(), dqci, {}, device_extensions,
                           &phys_dev_features, &phys_dev_features13);
  device = physical_device.createDevice(dci);
  vk::CommandPoolCreateInfo command_pool_ci(vk::CommandPoolCreateFlags(),
                                            c_qfi);
  command_pool = device.createCommandPool(command_pool_ci);
  queue = device.getQueue(c_qfi, 0);
  fence = device.createFence(vk::FenceCreateInfo());
  VmaAllocatorCreateInfo allocator_info{};
  allocator_info.physicalDevice = physical_device;
  allocator_info.vulkanApiVersion = physical_device.getProperties().apiVersion;
  allocator_info.device = device;
  allocator_info.instance = *instance;
  vmaCreateAllocator(&allocator_info, &allocator);
  vk::BufferCreateInfo staging_bci({}, round_up_x16(staging_size),
                                   vk::BufferUsageFlagBits::eTransferSrc |
                                       vk::BufferUsageFlagBits::eTransferDst);
  VmaAllocationCreateInfo alloc_create_info{};
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
  alloc_create_info.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  staging_allocation = VmaAllocation{};
  staging_info = VmaAllocationInfo{};
  vmaCreateBuffer(allocator, bit_cast<VkBufferCreateInfo*>(&staging_bci),
                  &alloc_create_info, bit_cast<VkBuffer*>(&staging),
                  &staging_allocation, &staging_info);
}

vk::CommandBuffer Manager::copy_op(vk::Buffer src_buffer, vk::Buffer dst_buffer,
                                   u32 buffer_size, u32 src_offset,
                                   u32 dst_offset) const {
  auto cmd_buffer = device
                        .allocateCommandBuffers(
                            {command_pool, vk::CommandBufferLevel::ePrimary, 1})
                        .front();
  vk::CommandBufferBeginInfo cbbi{};
  cmd_buffer.begin(cbbi);
  cmd_buffer.copyBuffer(src_buffer, dst_buffer,
                        vk::BufferCopy(src_offset, dst_offset, buffer_size));
  cmd_buffer.end();
  return cmd_buffer;
}

void Manager::copy_buffer(vk::Buffer& src_buffer, vk::Buffer& dst_buffer,
                          u32 buffer_size, u32 src_offset,
                          u32 dst_offset) const {
  copy_buffer_now(device, command_pool, queue, fence, src_buffer, dst_buffer,
                  buffer_size, src_offset, dst_offset);
}

vk::CommandBuffer
Manager::begin_record(vk::CommandBufferUsageFlagBits bits) const {
  auto cmd_buffer = device
                        .allocateCommandBuffers(
                            {command_pool, vk::CommandBufferLevel::ePrimary, 1})
                        .front();
  vk::CommandBufferBeginInfo cbbi(bits);
  cmd_buffer.begin(cbbi);

  return cmd_buffer;
}

void Manager::write_to_buffer(MetaBuffer& dest, const void* source, size_t size,
                              size_t src_offset, size_t dst_offset) {
  // Catch if we're trying to write more data than the staging buffer can
  // store.
  if (size > staging_info.size) {
    vmaDestroyBuffer(allocator, staging, staging_allocation);
    vk::BufferCreateInfo staging_bci({}, round_up_x16(size),
                                     vk::BufferUsageFlagBits::eTransferSrc |
                                         vk::BufferUsageFlagBits::eTransferDst);
    VmaAllocationCreateInfo alloc_create_info{};
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_create_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    staging_allocation = VmaAllocation{};
    staging_info = VmaAllocationInfo{};
    vmaCreateBuffer(allocator, bit_cast<VkBufferCreateInfo*>(&staging_bci),
                    &alloc_create_info, bit_cast<VkBuffer*>(&staging),
                    &staging_allocation, &staging_info);
  }

  memcpy(staging_info.pMappedData, source, size);
  copy_buffer(staging, dest.buffer, size, src_offset, dst_offset);
}

void Manager::write_from_buffer(MetaBuffer& source, void* dest, size_t size) {
  // Catch if we're trying to write more data than the staging buffer can
  // store.
  if (size > staging_info.size) {
    vmaDestroyBuffer(allocator, staging, staging_allocation);
    vk::BufferCreateInfo staging_bci({}, round_up_x16(size),
                                     vk::BufferUsageFlagBits::eTransferSrc |
                                         vk::BufferUsageFlagBits::eTransferDst);
    VmaAllocationCreateInfo alloc_create_info{};
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_create_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    staging_allocation = VmaAllocation{};
    staging_info = VmaAllocationInfo{};
    vmaCreateBuffer(allocator, bit_cast<VkBufferCreateInfo*>(&staging_bci),
                    &alloc_create_info, bit_cast<VkBuffer*>(&staging),
                    &staging_allocation, &staging_info);
  }

  copy_buffer(source.buffer, staging, size);
  memcpy(dest, staging_info.pMappedData, size);
}

void Manager::execute(vk::CommandBuffer& b) {
  vk::SubmitInfo submit_info(0, nullptr, nullptr, 1, &b);
  queue.submit(submit_info, fence);
  auto result = device.waitForFences(fence, vk::True, -1);
  result = device.resetFences(1, &fence);
}

void Manager::execute_no_sync(vk::CommandBuffer& b) const {
  vk::SubmitInfo submit_info(0, nullptr, nullptr, 1, &b);
  queue.submit(submit_info);
}

void Manager::queue_wait_idle() const { queue.waitIdle(); }

Algorithm Manager::make_algorithm(std::string spirvname,
                                  const std::vector<vk::ImageView>& images,
                                  const std::vector<MetaBuffer*>& buffers,
                                  std::span<const f32> spec_consts,
                                  size_t n_push_constants) const {
  const std::vector<u32> spirv = read_file<u32>(spirvname);
  auto retalg = Algorithm(device, spirv, images.size(), buffers.size(), 0,
                          spec_consts, n_push_constants);
  retalg.bind_data({images.begin(), images.end()}, buffers, {});
  return retalg;
}

// Algorithm Manager::make_algorithm_raw(std::string spirvname,
//                                       const std::vector<vk::ImageView>&
//                                       images, const std::vector<MetaBuffer*>&
//                                       buffers, const u8* spec_consts, const
//                                       size_t* spec_const_offsets, size_t
//                                       n_consts, const size_t* push_sizes,
//                                       size_t n_push_constants) const {
//   const auto spirv = read_file<u32>(spirvname);
//   auto retalg =
//       Algorithm(device, images.size(), buffers.size(), 0, spirv, spec_consts,
//                 spec_const_offsets, n_consts, push_sizes, n_push_constants);
//   retalg.bind_data(images, buffers, {});
//   return retalg;
// }

void append_op_no_barrier(vk::CommandBuffer b, const Algorithm& a, u32 x, u32 y,
                          u32 z) {
  b.bindPipeline(vk::PipelineBindPoint::eCompute, a.m_Pipeline);
  b.bindDescriptorSets(vk::PipelineBindPoint::eCompute, a.m_PipelineLayout, 0,
                       a.m_DescriptorSet, nullptr);
  b.dispatch(x, y, z);
}

void append_op(vk::CommandBuffer b, const Algorithm& a, u32 x, u32 y, u32 z) {
  b.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                    vk::PipelineStageFlagBits::eAllCommands, {},
                    FULL_MEMORY_BARRIER, nullptr, nullptr);
  b.bindPipeline(vk::PipelineBindPoint::eCompute, a.m_Pipeline);
  b.bindDescriptorSets(vk::PipelineBindPoint::eCompute, a.m_PipelineLayout, 0,
                       a.m_DescriptorSet, nullptr);
  b.dispatch(x, y, z);
}

Manager::~Manager() {
  device.waitIdle();
  device.destroyFence(fence);
  vmaDestroyBuffer(allocator, staging, staging_allocation);
  vmaDestroyAllocator(allocator);
  device.destroyCommandPool(command_pool);
  device.destroy();
}

void Algorithm::initialize(vk::Device device, std::span<const u32> spirv,
                           u32 n_imgs, u32 n_buffers, u32 n_ubo,
                           std::span<const f32> spec_consts,
                           size_t n_push_constants) {
  m_device = device;
  vk::ShaderModuleCreateInfo shader_mci(vk::ShaderModuleCreateFlags(), spirv);
  m_ShaderModule = device.createShaderModule(shader_mci);
  std::vector<vk::DescriptorSetLayoutBinding> dslbs(n_imgs + n_buffers + n_ubo);
  {
    u32 i = 0;
    for (; i < n_imgs; i++) {
      dslbs[i] = {i, vk::DescriptorType::eStorageImage, 1,
                  vk::ShaderStageFlagBits::eCompute};
    }
    for (; i < n_buffers + n_imgs; i++) {
      dslbs[i] = {i, vk::DescriptorType::eStorageBuffer, 1,
                  vk::ShaderStageFlagBits::eCompute};
    }
    for (; i < n_buffers + n_imgs + n_ubo; i++) {
      dslbs[i] = {i, vk::DescriptorType::eUniformBuffer, 1,
                  vk::ShaderStageFlagBits::eCompute};
    }
  }
  vk::DescriptorSetLayoutCreateInfo dslci(vk::DescriptorSetLayoutCreateFlags(),
                                          dslbs);
  m_DSL = device.createDescriptorSetLayout(dslci);
  std::vector<vk::PushConstantRange> ranges(n_push_constants);
  u32 push_offsets = 0;
  for (u32 i = 0; i < n_push_constants; i++) {
    ranges[i] = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute,
                                      push_offsets, 4);
    push_offsets += 4;
  }
  vk::PipelineLayoutCreateInfo plci(vk::PipelineLayoutCreateFlags(), m_DSL,
                                    ranges);
  m_PipelineLayout = device.createPipelineLayout(plci);
  std::vector<vk::SpecializationMapEntry> spec_entries(spec_consts.size());
  u32 spec_offset = 0;
  for (u32 i = 0; i < spec_entries.size(); ++i) {
    spec_entries[i].constantID = i;
    spdlog::debug("Setting spec_entries[{}].offset to {}", i, spec_offset);
    spec_entries[i].offset = spec_offset;
    spec_entries[i].size = 4;
    spec_offset += 4;
  }
  vk::SpecializationInfo spec_info;
  spec_info.mapEntryCount = spec_consts.size();
  spec_info.pMapEntries = spec_entries.data();
  spec_info.dataSize = spec_offset;
  spec_info.pData = spec_consts.data();

  vk::PipelineShaderStageCreateInfo csci(vk::PipelineShaderStageCreateFlags(),
                                         vk::ShaderStageFlagBits::eCompute,
                                         m_ShaderModule, "main", &spec_info);
  vk::ComputePipelineCreateInfo cpci(vk::PipelineCreateFlags(), csci,
                                     m_PipelineLayout);
  auto result = device.createComputePipeline({}, cpci);
  m_Pipeline = result.value;

  // This is probably not the most efficient way to do this, but I'm not going
  // to mess around with the descriptors after creation so the only overhead
  // should be memory, and I'm not going to make thousands of these so
  // it should be fine.
  std::vector<vk::DescriptorPoolSize> dpss;
  if (n_imgs > 0) {
    dpss.emplace_back(vk::DescriptorType::eStorageImage, n_imgs);
  }
  if (n_buffers > 0) {
    dpss.emplace_back(vk::DescriptorType::eStorageBuffer, n_buffers);
  }
  if (n_ubo > 0) {
    dpss.emplace_back(vk::DescriptorType::eUniformBuffer, n_ubo);
  }
  vk::DescriptorPoolCreateInfo dpci(
      vk::DescriptorPoolCreateFlags(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
      1, dpss);
  m_DescriptorPool = device.createDescriptorPool(dpci);
  vk::DescriptorSetAllocateInfo dsai(m_DescriptorPool, 1, &m_DSL);
  auto descriptor_sets = device.allocateDescriptorSets(dsai);
  m_DescriptorSet = descriptor_sets[0];
}

void Algorithm::bind_data(std::span<const vk::ImageView> img_views,
                          std::span<const MetaBuffer* const> buffers,
                          std::span<const MetaBuffer* const> ubos) const {
  std::vector<vk::DescriptorImageInfo> diis(img_views.size());
  std::vector<vk::DescriptorBufferInfo> dbis(buffers.size());
  std::vector<vk::DescriptorBufferInfo> dubis(ubos.size());
  for (size_t i = 0; i < diis.size(); i++) {
    diis[i] = {{}, img_views[i], vk::ImageLayout::eGeneral};
  }
  for (size_t i = 0; i < dbis.size(); i++) {
    dbis[i] =
        vk::DescriptorBufferInfo(buffers[i]->buffer, 0, buffers[i]->aInfo.size);
  }
  for (size_t i = 0; i < dubis.size(); i++) {
    dbis[i] = vk::DescriptorBufferInfo(ubos[i]->buffer, 0, ubos[i]->aInfo.size);
  }

  std::vector<vk::WriteDescriptorSet> write_descriptor_sets(
      diis.size() + dbis.size() + dubis.size());
  {
    u32 i = 0;
    for (; i < diis.size(); i++) {
      write_descriptor_sets[i] = {m_DescriptorSet, i, 0,
                                  vk::DescriptorType::eStorageImage, diis[i]};
    }
    for (; i < diis.size() + dbis.size(); i++) {
      write_descriptor_sets[i] = {m_DescriptorSet,
                                  i,
                                  0,
                                  1,
                                  vk::DescriptorType::eStorageBuffer,
                                  nullptr,
                                  &dbis[i - diis.size()]};
    }
    for (; i < diis.size() + dbis.size() + dubis.size(); i++) {
      write_descriptor_sets[i] = {m_DescriptorSet,
                                  i,
                                  0,
                                  1,
                                  vk::DescriptorType::eUniformBuffer,
                                  nullptr,
                                  &dubis[i - diis.size() - dbis.size()]};
    }
  }
  m_device.updateDescriptorSets(write_descriptor_sets, {});
}

void Renderer::make_depth_img_and_view() {
  depth_fmt = pick_depth_format(physical_device);
  depth_img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  depth_img_ci.imageType = VK_IMAGE_TYPE_2D;
  depth_img_ci.format = depth_fmt;
  depth_img_ci.extent.width = swapchain_extent.width;
  depth_img_ci.extent.height = swapchain_extent.height;
  depth_img_ci.extent.depth = 1;
  depth_img_ci.mipLevels = 1;
  depth_img_ci.arrayLayers = 1;
  depth_img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
  depth_img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  depth_img_ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  depth_img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo alloc_ci{};
  alloc_ci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
  chk(vmaCreateImage(allocator, &depth_img_ci, &alloc_ci, &depth_img,
                     &depth_allocation, nullptr));
  VkImageViewCreateInfo depth_view_ci{};
  depth_view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
  depth_view_ci.image = depth_img,
  depth_view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D,
  depth_view_ci.format = depth_fmt,
  depth_view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  depth_view_ci.subresourceRange.levelCount = 1;
  depth_view_ci.subresourceRange.layerCount = 1;
  chk(vkCreateImageView(device, &depth_view_ci, nullptr, &depth_view));
}

void Renderer::recreate_swapchain() {

  DEBUG_START;
  s32 win_width = 0;
  s32 win_height = 0;
  chk_sdl(SDL_GetWindowSize(window, &win_width, &win_height));
  chk(vkDeviceWaitIdle(device));
  chk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,
                                                &surface_caps));
  DEBUG_LOG("Got window size, waited for device and got surface capabilities");
  swapchain_ci.oldSwapchain = swapchain;
  swapchain_ci.imageExtent = swapchain_extent;

  chk(vkCreateSwapchainKHR(device, &swapchain_ci, nullptr, &swapchain));
  DEBUG_LOG("Ran CreateSwapchain");
  u32 image_count = swapchain_imgs.size();
  for (u32 i = 0; i < image_count; ++i) {
    vkDestroyImageView(device, swapchain_img_views[i], nullptr);
  }
  DEBUG_LOG("getting number of swapchain images");
  chk(vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr));
  swapchain_imgs.resize(image_count);
  DEBUG_LOG("getting swapchain images");
  chk(vkGetSwapchainImagesKHR(device, swapchain, &image_count,
                              swapchain_imgs.data()));
  // swapchain_imgs = get_swapchain_imgs(device, swapchain);
  DEBUG_LOG("resizing swapchain image views");
  swapchain_img_views.resize(image_count);
  for (u32 i = 0; i < image_count; i++) {
    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = swapchain_imgs[i];
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = SWAPCHAIN_IMAGE_FORMAT;
    view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_ci.subresourceRange.levelCount = 1;
    view_ci.subresourceRange.layerCount = 1;
    DEBUG_LOG("Creating swapchain image view");
    chk(vkCreateImageView(device, &view_ci, nullptr, &swapchain_img_views[i]));
  }
  for (auto& semaphore : render_complete_semaphores) {
    DEBUG_LOG("Destroying semaphore");
    vkDestroySemaphore(device, semaphore, nullptr);
  }
  render_complete_semaphores.resize(image_count);
  VkSemaphoreCreateInfo semaphore_ci{};
  for (auto& semaphore : render_complete_semaphores) {
    DEBUG_LOG("Creating semaphore");
    chk(vkCreateSemaphore(device, &semaphore_ci, nullptr, &semaphore));
  }
  DEBUG_LOG("Destroying old swapchain");
  vkDestroySwapchainKHR(device, swapchain_ci.oldSwapchain, nullptr);
  DEBUG_LOG("Destroying old depth image");
  vmaDestroyImage(allocator, depth_img, depth_allocation);
  DEBUG_LOG("Destroying old depth image view");
  vkDestroyImageView(device, depth_view, nullptr);
  depth_img_ci.extent = {.width = swapchain_extent.width,
                         .height = swapchain_extent.height,
                         .depth = 1};
  VmaAllocationCreateInfo alloc_ci{};
  alloc_ci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
  DEBUG_LOG("Creating new depth image");
  chk(vmaCreateImage(allocator, &depth_img_ci, &alloc_ci, &depth_img,
                     &depth_allocation, nullptr));
  VkImageViewCreateInfo view_ci{};
  view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_ci.image = depth_img;
  view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_ci.format = depth_fmt;
  view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  view_ci.subresourceRange.levelCount = 1;
  view_ci.subresourceRange.layerCount = 1;
  DEBUG_LOG("Creating new depth image view");
  chk(vkCreateImageView(device, &view_ci, nullptr, &depth_view));
  DEBUG_END;
}

void Renderer::init_sync_objects() {
  DEBUG_START;
  VkSemaphoreCreateInfo semaphore_ci{};
  semaphore_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fence_ci{};
  fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    chk(vkCreateFence(device, &fence_ci, nullptr, &fences[i]));
    chk(vkCreateSemaphore(device, &semaphore_ci, nullptr,
                          &img_acquired_semaphores[i]));
  }
  render_complete_semaphores.resize(swapchain_imgs.size());
  for (auto& semaphore : render_complete_semaphores) {
    chk(vkCreateSemaphore(device, &semaphore_ci, nullptr, &semaphore));
  }
  DEBUG_END;
}

VkDescriptorImageInfo Renderer::load_tex_img() {
  DEBUG_START;

  s32 tex_width = 0;
  s32 tex_height = 0;
  s32 tex_channels = 0;
  stbi_uc* pixels = stbi_load("assets/kyousaya.jpg", &tex_width, &tex_height,
                              &tex_channels, STBI_rgb_alpha);
  VkDeviceSize image_size = static_cast<s64>(tex_width) * tex_height * 4;
  if (!static_cast<bool>(pixels)) {
    std::cerr << "Failed to load texture image\n";
    exit(-1);
  }
  u32 width = tex_width;
  u32 height = tex_height;
  VkImageCreateInfo tex_img_ci{};
  tex_img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  tex_img_ci.imageType = VK_IMAGE_TYPE_2D;
  tex_img_ci.format = VK_FORMAT_R8G8B8A8_SRGB;
  tex_img_ci.extent.width = width;
  tex_img_ci.extent.height = height;
  tex_img_ci.extent.depth = 1;
  tex_img_ci.mipLevels = 1;
  tex_img_ci.arrayLayers = 1;
  tex_img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
  tex_img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  tex_img_ci.usage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  tex_img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo tex_image_alloc_ci{};
  tex_image_alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
  chk(vmaCreateImage(allocator, &tex_img_ci, &tex_image_alloc_ci,
                     &texture.image, &texture.allocation, nullptr));

  VkImageViewCreateInfo tex_view_ci{};
  tex_view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  tex_view_ci.image = texture.image;
  tex_view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  tex_view_ci.format = tex_img_ci.format;
  tex_view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  tex_view_ci.subresourceRange.levelCount = 1;
  tex_view_ci.subresourceRange.layerCount = 1;
  chk(vkCreateImageView(device, &tex_view_ci, nullptr, &texture.view));

  VkBuffer img_src_buffer{};
  VmaAllocation img_src_allocation{};
  VkBufferCreateInfo img_src_buffer_ci{};
  img_src_buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  img_src_buffer_ci.size = image_size;
  img_src_buffer_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VmaAllocationCreateInfo img_src_alloc_ci{};
  img_src_alloc_ci.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  img_src_alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
  VmaAllocationInfo img_src_alloc_info{};
  chk(vmaCreateBuffer(allocator, &img_src_buffer_ci, &img_src_alloc_ci,
                      &img_src_buffer, &img_src_allocation,
                      &img_src_alloc_info));

  memcpy(img_src_alloc_info.pMappedData, pixels, image_size);
  VkFenceCreateInfo fence_one_time_ci{};
  fence_one_time_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence_one_time{};
  chk(vkCreateFence(device, &fence_one_time_ci, nullptr, &fence_one_time));
  VkCommandBuffer cb_one_time{};
  VkCommandBufferAllocateInfo cb_one_time_ai{};
  cb_one_time_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
  cb_one_time_ai.commandPool = command_pool,
  cb_one_time_ai.commandBufferCount = 1;
  chk(vkAllocateCommandBuffers(device, &cb_one_time_ai, &cb_one_time));
  VkCommandBufferBeginInfo cb_one_time_bi{};
  cb_one_time_bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  cb_one_time_bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  chk(vkBeginCommandBuffer(cb_one_time, &cb_one_time_bi));
  VkImageMemoryBarrier2 barrier_tex_image{};
  barrier_tex_image.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  barrier_tex_image.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
  barrier_tex_image.srcAccessMask = VK_ACCESS_2_NONE;
  barrier_tex_image.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier_tex_image.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier_tex_image.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier_tex_image.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier_tex_image.image = texture.image;
  barrier_tex_image.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier_tex_image.subresourceRange.levelCount = 1;
  barrier_tex_image.subresourceRange.layerCount = 1;
  VkDependencyInfo barrier_tex_info{};
  barrier_tex_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  barrier_tex_info.imageMemoryBarrierCount = 1;
  barrier_tex_info.pImageMemoryBarriers = &barrier_tex_image;
  vkCmdPipelineBarrier2(cb_one_time, &barrier_tex_info);
  VkBufferImageCopy region{};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {.width = width, .height = height, .depth = 1};
  vkCmdCopyBufferToImage(cb_one_time, img_src_buffer, texture.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  VkImageMemoryBarrier2 barrier_tex_read{};
  barrier_tex_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  barrier_tex_read.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
  barrier_tex_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier_tex_read.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  barrier_tex_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier_tex_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier_tex_read.newLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
  barrier_tex_read.image = texture.image;
  barrier_tex_read.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier_tex_read.subresourceRange.levelCount = 1;
  barrier_tex_read.subresourceRange.layerCount = 1;
  barrier_tex_info.pImageMemoryBarriers = &barrier_tex_read;
  vkCmdPipelineBarrier2(cb_one_time, &barrier_tex_info);
  chk(vkEndCommandBuffer(cb_one_time));
  VkSubmitInfo one_time_si{};
  one_time_si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  one_time_si.commandBufferCount = 1;
  one_time_si.pCommandBuffers = &cb_one_time;
  chk(vkQueueSubmit(queue, 1, &one_time_si, fence_one_time));
  chk(vkWaitForFences(device, 1, &fence_one_time, VK_TRUE, UINT64_MAX));
  vkDestroyFence(device, fence_one_time, nullptr);
  vmaDestroyBuffer(allocator, img_src_buffer, img_src_allocation);
  // Sampler
  VkSamplerCreateInfo sampler_ci{};
  sampler_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_ci.magFilter = VK_FILTER_LINEAR;
  sampler_ci.minFilter = VK_FILTER_LINEAR;
  sampler_ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_ci.anisotropyEnable = VK_TRUE;
  sampler_ci.maxAnisotropy = 8.0F;
  chk(vkCreateSampler(device, &sampler_ci, nullptr, &texture.sampler));

  VkDescriptorImageInfo texture_descriptor{};
  texture_descriptor.sampler = texture.sampler;
  texture_descriptor.imageView = texture.view,
  texture_descriptor.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
  DEBUG_END;
  return texture_descriptor;
}

VkShaderModule Renderer::load_main_shader() const {
  DEBUG_START;
  auto slang_targets{std::to_array<slang::TargetDesc>(
      {{.format = SLANG_SPIRV,
        .profile = slang_global_session->findProfile("spirv_1_4")}})};
  // auto slang_options{std::to_array<slang::CompilerOptionEntry>(
  //     {{slang::CompilerOptionName::EmitSpirvDirectly,
  //       {slang::CompilerOptionValueKind::Int, 1}}})};
  slang::CompilerOptionEntry entry{};
  entry.name = slang::CompilerOptionName::EmitSpirvDirectly;
  entry.value.kind = slang::CompilerOptionValueKind::Int;
  entry.value.intValue0 = 1;
  slang::SessionDesc slang_session_desc{
      .targets = slang_targets.data(),
      .targetCount = SlangInt(slang_targets.size()),
      .defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
      .compilerOptionEntries = &entry, // slang_options.data(),
      .compilerOptionEntryCount = 1};

  Slang::ComPtr<slang::ISession> slang_session;
  slang_global_session->createSession(slang_session_desc,
                                      slang_session.writeRef());
  Slang::ComPtr<slang::IModule> slang_module{
      slang_session->loadModuleFromSource("triangle", "shaders/shader.slang",
                                          nullptr, nullptr)};
  Slang::ComPtr<ISlangBlob> spirv;
  slang_module->getTargetCode(0, spirv.writeRef());
  VkShaderModuleCreateInfo shader_module_ci{};
  shader_module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
  shader_module_ci.codeSize = spirv->getBufferSize(),
  shader_module_ci.pCode = (uint32_t*)spirv->getBufferPointer();
  VkShaderModule shader_module{};
  chk(vkCreateShaderModule(device, &shader_module_ci, nullptr, &shader_module));
  DEBUG_END;
  return shader_module;
}

Renderer::Renderer(SDL_Window* window, VkSurfaceKHR surf) {
  DEBUG_START;
  chk_sdl(SDL_Init(SDL_INIT_VIDEO));
  chk_sdl(SDL_Vulkan_LoadLibrary(nullptr));
  volkInitialize();
  // Instance
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "MyRenderer";
  app_info.apiVersion = VK_API_VERSION_1_3;
  u32 instance_extensions_count = 0;
  char const* const* instance_extensions{
      SDL_Vulkan_GetInstanceExtensions(&instance_extensions_count)};
  std::array<const char*, 1> validation_layers{"VK_LAYER_KHRONOS_validation"};
  VkInstanceCreateInfo instance_ci{};
  instance_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_ci.pApplicationInfo = &app_info;
  instance_ci.enabledExtensionCount = instance_extensions_count;
  instance_ci.ppEnabledExtensionNames = instance_extensions;
  instance_ci.enabledLayerCount = 1;
  instance_ci.ppEnabledLayerNames = validation_layers.data();
  chk(vkCreateInstance(&instance_ci, nullptr, &instance));
  volkLoadInstance(instance);
  physical_device = pick_physical_device(instance);
  VkPhysicalDeviceProperties2 dev_props{};
  dev_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  vkGetPhysicalDeviceProperties2(physical_device, &dev_props);
  std::cout << "Selected device: " << dev_props.properties.deviceName << '\n';
  // Find a queue family for graphics
  uint32_t queue_family_count{0};
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count,
                                           queue_families.data());
  uint32_t queue_family{0};
  for (size_t i = 0; i < queue_families.size(); i++) {
    if (static_cast<bool>(queue_families[i].queueFlags &
                          VK_QUEUE_GRAPHICS_BIT)) {
      queue_family = i;
      break;
    }
  }
  DEBUG_LOG("Queue family should be: {}", queue_family);
  chk_sdl(SDL_Vulkan_GetPresentationSupport(instance, physical_device,
                                            queue_family));
  device = make_device(physical_device, {&queue_family, size_t{1}}, true);
  vkGetDeviceQueue(device, queue_family, 0, &queue);
  // VMA
  VmaVulkanFunctions vk_functions{};
  vk_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vk_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
  vk_functions.vkCreateImage = vkCreateImage;

  VmaAllocatorCreateInfo allocator_ci{};
  allocator_ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  allocator_ci.physicalDevice = physical_device;
  allocator_ci.vulkanApiVersion = VK_API_VERSION_1_3;
  allocator_ci.device = device;
  allocator_ci.pVulkanFunctions = &vk_functions;
  allocator_ci.instance = instance;
  chk(vmaCreateAllocator(&allocator_ci, &allocator));
  DEBUG_LOG("Created allocator");
  // Window and surface
  window = SDL_CreateWindow("How to Vulkan", 1280U, 720U,
                            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
  assert(window);
  chk_sdl(SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface));
  s32 win_width = 0;
  s32 win_height = 0;
  chk_sdl(SDL_GetWindowSize(window, &win_width, &win_height));
  // swapchain_extent.width = static_cast<u32>(win_width);
  // swapchain_extent.height = static_cast<u32>(win_width);
  chk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,
                                                &surface_caps));
  swapchain_extent = surface_caps.currentExtent;
  if (surface_caps.currentExtent.width == 0xFFFFFFFF) {
    swapchain_extent = {.width = static_cast<uint32_t>(win_width),
                        .height = static_cast<uint32_t>(win_height)};
  }
  // Swap chain
  // const VkFormat imageFormat{VK_FORMAT_B8G8R8A8_SRGB};
  swapchain_ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapchain_ci.surface = surface;
  swapchain_ci.minImageCount = surface_caps.minImageCount;
  swapchain_ci.imageFormat = SWAPCHAIN_IMAGE_FORMAT;
  swapchain_ci.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
  swapchain_ci.imageExtent.width = swapchain_extent.width;
  swapchain_ci.imageExtent.height = swapchain_extent.height;
  swapchain_ci.imageArrayLayers = 1;
  swapchain_ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  swapchain_ci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  swapchain_ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchain_ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  chk(vkCreateSwapchainKHR(device, &swapchain_ci, nullptr, &swapchain));

  swapchain_imgs = get_swapchain_imgs(device, swapchain);
  u32 image_count = swapchain_imgs.size();
  swapchain_img_views.resize(image_count);
  for (u32 i = 0; i < image_count; i++) {
    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    view_ci.image = swapchain_imgs[i], view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D,
    view_ci.format = SWAPCHAIN_IMAGE_FORMAT,
    view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    view_ci.subresourceRange.levelCount = 1,
    view_ci.subresourceRange.layerCount = 1;
    chk(vkCreateImageView(device, &view_ci, nullptr, &swapchain_img_views[i]));
  }
  make_depth_img_and_view();

  init_sync_objects();
  // Command pool
  VkCommandPoolCreateInfo command_pool_ci{};
  command_pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
  command_pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
  command_pool_ci.queueFamilyIndex = queue_family;
  chk(vkCreateCommandPool(device, &command_pool_ci, nullptr, &command_pool));
  VkCommandBufferAllocateInfo cb_alloc_ci{};
  cb_alloc_ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
  cb_alloc_ci.commandPool = command_pool,
  cb_alloc_ci.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
  chk(vkAllocateCommandBuffers(device, &cb_alloc_ci, cbs.data()));
  // Texture images
  VkDescriptorImageInfo texture_descriptor = load_tex_img();
  // Descriptor (indexing)
  VkDescriptorBindingFlags desc_variable_flag{
      VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT};
  VkDescriptorSetLayoutBindingFlagsCreateInfo desc_binding_flags{};
  desc_binding_flags.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  desc_binding_flags.bindingCount = 1;
  desc_binding_flags.pBindingFlags = &desc_variable_flag;
  VkDescriptorSetLayoutBinding desc_layout_binding_tex{};
  desc_layout_binding_tex.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  desc_layout_binding_tex.descriptorCount = 1;
  // static_cast<uint32_t>(textures.size());
  desc_layout_binding_tex.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorSetLayoutCreateInfo desc_layout_tex_ci{};
  desc_layout_tex_ci.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  desc_layout_tex_ci.pNext = &desc_binding_flags;
  desc_layout_tex_ci.bindingCount = 1;
  desc_layout_tex_ci.pBindings = &desc_layout_binding_tex;
  chk(vkCreateDescriptorSetLayout(device, &desc_layout_tex_ci, nullptr,
                                  &descriptor_set_layout_tex));
  VkDescriptorPoolSize pool_size{
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1}; // static_cast<uint32_t>(textures.size())};
  VkDescriptorPoolCreateInfo desc_pool_ci{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = {},
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size};
  chk(vkCreateDescriptorPool(device, &desc_pool_ci, nullptr, &descriptor_pool));
  uint32_t variable_desc_count = 1; // {static_cast<uint32_t>(textures.size())};
  VkDescriptorSetVariableDescriptorCountAllocateInfo variable_desc_count_ai{};
  variable_desc_count_ai.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT;
  variable_desc_count_ai.descriptorSetCount = 1;
  variable_desc_count_ai.pDescriptorCounts = &variable_desc_count;

  VkDescriptorSetAllocateInfo tex_desc_set_alloc{};
  tex_desc_set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  tex_desc_set_alloc.pNext = &variable_desc_count_ai;
  tex_desc_set_alloc.descriptorPool = descriptor_pool;
  tex_desc_set_alloc.descriptorSetCount = 1;
  tex_desc_set_alloc.pSetLayouts = &descriptor_set_layout_tex;
  chk(vkAllocateDescriptorSets(device, &tex_desc_set_alloc,
                               &descriptor_set_tex));
  VkWriteDescriptorSet write_desc_set{};
  write_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_desc_set.dstSet = descriptor_set_tex;
  write_desc_set.dstBinding = 0;
  write_desc_set.descriptorCount = 1;
  // static_cast<uint32_t>(texture_descriptors.size());
  write_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write_desc_set.pImageInfo =
      &texture_descriptor; // texture_descriptors.data();
  vkUpdateDescriptorSets(device, 1, &write_desc_set, 0, nullptr);
  // Initialize Slang shader compiler
  slang::createGlobalSession(slang_global_session.writeRef());
  VkShaderModule shader_module = load_main_shader();

  VkPipelineLayoutCreateInfo pipeline_layout_ci{};
  pipeline_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
  pipeline_layout_ci.setLayoutCount = 1,
  pipeline_layout_ci.pSetLayouts = &descriptor_set_layout_tex,
  chk(vkCreatePipelineLayout(device, &pipeline_layout_ci, nullptr,
                             &pipeline_layout));
  std::vector<VkPipelineShaderStageCreateInfo> shader_stages(2);
  shader_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
  shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT,
  shader_stages[0].module = shader_module, shader_stages[0].pName = "main";

  shader_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  shader_stages[1].module = shader_module;
  shader_stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo vertex_input_state{};
  vertex_input_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input_state.vertexBindingDescriptionCount = 0;
  vertex_input_state.vertexAttributeDescriptionCount = 0;

  VkPipelineInputAssemblyStateCreateInfo input_assembly_state{};
  input_assembly_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  std::vector<VkDynamicState> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT,
                                             VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dynamic_states.data();

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterization_state{};
  rasterization_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization_state.lineWidth = 1.0F;

  VkPipelineMultisampleStateCreateInfo multisample_state{};
  multisample_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depth_stencil_state{};
  depth_stencil_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil_state.depthTestEnable = VK_TRUE;
  depth_stencil_state.depthWriteEnable = VK_TRUE;
  depth_stencil_state.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

  VkPipelineColorBlendAttachmentState blend_attachment{};
  blend_attachment.colorWriteMask = 0xF;

  VkPipelineColorBlendStateCreateInfo color_blend_state{};
  color_blend_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blend_state.attachmentCount = 1;
  color_blend_state.pAttachments = &blend_attachment;

  VkPipelineRenderingCreateInfo rendering_ci{};
  rendering_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
  rendering_ci.colorAttachmentCount = 1;
  rendering_ci.pColorAttachmentFormats = &SWAPCHAIN_IMAGE_FORMAT;
  rendering_ci.depthAttachmentFormat = depth_fmt;

  VkGraphicsPipelineCreateInfo pipeline_ci{};
  pipeline_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_ci.pNext = &rendering_ci;
  pipeline_ci.stageCount = 2;
  pipeline_ci.pStages = shader_stages.data();
  pipeline_ci.pVertexInputState = &vertex_input_state;
  pipeline_ci.pInputAssemblyState = &input_assembly_state;
  pipeline_ci.pViewportState = &viewport_state;
  pipeline_ci.pRasterizationState = &rasterization_state;
  pipeline_ci.pMultisampleState = &multisample_state;
  pipeline_ci.pDepthStencilState = &depth_stencil_state;
  pipeline_ci.pColorBlendState = &color_blend_state;
  pipeline_ci.pDynamicState = &dynamic_state;
  pipeline_ci.layout = pipeline_layout;
  chk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci,
                                nullptr, &pipeline));
}

void Renderer::draw_frame() {
  DEBUG_START;
  DEBUG_LOG("Waiting for fences");
  chk(vkWaitForFences(device, 1, &fences[frame_index], VK_TRUE, UINT64_MAX));
  DEBUG_LOG("Acquiring swapchain images");
  u32 image_index = 0;
  if (!chk_swapchain(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                           img_acquired_semaphores[frame_index],
                                           VK_NULL_HANDLE, &image_index))) {
    recreate_swapchain();
    return;
  }
  chk(vkResetFences(device, 1, &fences[frame_index]));
  // Build command buffer
  auto* cb = cbs[frame_index];
  chk(vkResetCommandBuffer(cb, 0));
  VkCommandBufferBeginInfo cb_bi{};
  cb_bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  cb_bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  chk(vkBeginCommandBuffer(cb, &cb_bi));
  std::array<VkImageMemoryBarrier2, 2> output_barriers{};
  output_barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  output_barriers[0].srcStageMask =
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  output_barriers[0].srcAccessMask = 0;
  output_barriers[0].dstStageMask =
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  output_barriers[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  output_barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  output_barriers[0].newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
  output_barriers[0].image = swapchain_imgs[image_index];
  output_barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  output_barriers[0].subresourceRange.levelCount = 1;
  output_barriers[0].subresourceRange.layerCount = 1;

  output_barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
  output_barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
  output_barriers[1].srcAccessMask =
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
  output_barriers[1].dstStageMask =
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
  output_barriers[1].dstAccessMask =
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
  output_barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  output_barriers[1].newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
  output_barriers[1].image = depth_img,
  output_barriers[1].subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
  output_barriers[1].subresourceRange.levelCount = 1,
  output_barriers[1].subresourceRange.layerCount = 1;
  VkDependencyInfo barrier_dependency_info{};
  barrier_dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  barrier_dependency_info.imageMemoryBarrierCount = 2;
  barrier_dependency_info.pImageMemoryBarriers = output_barriers.data();
  vkCmdPipelineBarrier2(cb, &barrier_dependency_info);
  VkRenderingAttachmentInfo color_attachment_info{};
  color_attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
  color_attachment_info.imageView = swapchain_img_views[image_index],
  color_attachment_info.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
  color_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
  color_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
  color_attachment_info.clearValue.color = {{0.0F, 0.0F, 0.0F, 1.0F}};
  VkRenderingAttachmentInfo depth_attachment_info{};
  depth_attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
  depth_attachment_info.imageView = depth_view,
  depth_attachment_info.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
  depth_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
  depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
  depth_attachment_info.clearValue.depthStencil = {.depth = 1.0F, .stencil = 0};
  VkRenderingInfo rendering_info{};
  rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
  rendering_info.renderArea.extent = swapchain_extent;
  rendering_info.layerCount = 1;
  rendering_info.colorAttachmentCount = 1;
  rendering_info.pColorAttachments = &color_attachment_info,
  rendering_info.pDepthAttachment = &depth_attachment_info;
  vkCmdBeginRendering(cb, &rendering_info);
  VkViewport vp{};
  vp.width = static_cast<f32>(swapchain_extent.width);
  vp.height = static_cast<f32>(swapchain_extent.height);
  vp.minDepth = 0.0F;
  vp.maxDepth = 1.0F;
  vkCmdSetViewport(cb, 0, 1, &vp);
  VkRect2D scissor{};
  scissor.extent = swapchain_extent;
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  vkCmdSetScissor(cb, 0, 1, &scissor);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                          0, 1, &descriptor_set_tex, 0, nullptr);
  vkCmdDraw(cb, 6, 1, 0, 0);
  vkCmdEndRendering(cb);
  VkImageMemoryBarrier2 barrier_present{};
  barrier_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  barrier_present.srcStageMask =
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  barrier_present.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  barrier_present.dstStageMask =
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  barrier_present.dstAccessMask = 0;
  barrier_present.oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
  barrier_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  barrier_present.image = swapchain_imgs[image_index];
  barrier_present.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier_present.subresourceRange.levelCount = 1;
  barrier_present.subresourceRange.layerCount = 1;
  VkDependencyInfo barrier_present_dependency_info{};
  barrier_present_dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  barrier_present_dependency_info.imageMemoryBarrierCount = 1;
  barrier_present_dependency_info.pImageMemoryBarriers = &barrier_present;
  vkCmdPipelineBarrier2(cb, &barrier_present_dependency_info);
  chk(vkEndCommandBuffer(cb));
  // Submit to graphics queue
  VkPipelineStageFlags wait_stages =
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = &img_acquired_semaphores[frame_index];
  submit_info.pWaitDstStageMask = &wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cb;
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &render_complete_semaphores[image_index];
  chk(vkQueueSubmit(queue, 1, &submit_info, fences[frame_index]));

  frame_index = (frame_index + 1) % MAX_FRAMES_IN_FLIGHT;

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &render_complete_semaphores[image_index];
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &swapchain;
  present_info.pImageIndices = &image_index;
  DEBUG_LOG("Presenting swapchain image {}", image_index);
  if (!chk_swapchain(vkQueuePresentKHR(queue, &present_info))) {
    recreate_swapchain();
  };
}
