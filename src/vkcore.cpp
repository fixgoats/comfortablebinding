#include "vkcore.hpp"
// #include "colormaps.h"
#include "mathhelpers.h"
#include "vulkan/vulkan.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
// #include <ranges>
#include "hack.hpp"
#include <set>

namespace {
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
    if (queue_family_props[i].queueFlags & vk::QueueFlagBits::eGraphics) {
      g_qfi = i;
    }
    if (static_cast<bool>(phys_dev.getSurfaceSupportKHR(i, surface))) {
      p_qfi = i;
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
  /*vk::CommandBufferAllocateInfo cb_alloc_info(
      command_pool, vk::CommandBufferLevel::ePrimary, n_images);
  command_buffers.resize(n_images);
  vkAllocateCommandBuffers(dev,
                           pcast<VkCommandBufferAllocateInfo>(&cb_alloc_info),
                           pcast<VkCommandBuffer>(command_buffers.data()));*/
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

// constexpr vk::SpecializationInfo
// span_to_spec_consts(std::span<const f32> spec_consts) {
//   std::vector<vk::SpecializationMapEntry> spec_entries(spec_consts.size());
//   u32 spec_offset = 0;
//   for (u32 i = 0; i < spec_entries.size(); i++) {
//     spec_entries[i].constantID = i;
//     spdlog::debug("Setting spec_entries[{}].offset to {}", i, spec_offset);
//     spec_entries[i].offset = spec_offset;
//     spec_entries[i].size = 4;
//     spec_offset += 4;
//   }
//   vk::SpecializationInfo spec_info;
//   spec_info.mapEntryCount = spec_consts.size();
//   spec_info.pMapEntries = spec_entries.data();
//   spec_info.dataSize = spec_offset;
//   spec_info.pData = spec_consts.data();
//   return spec_info;
// }

void record_nondestructive_parallel_reduction(
    vk::CommandBuffer cb, u32 org_size, const Algorithm& reduction,
    const Algorithm& first_reduction) {
  append_op(cb, first_reduction, org_size / (64 * 2), 1, 1);
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                     vk::PipelineStageFlagBits::eAllCommands, {},
                     FULL_MEMORY_BARRIER, nullptr, nullptr);
  cb.bindPipeline(vk::PipelineBindPoint::eCompute, reduction.m_Pipeline);
  cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                        reduction.m_PipelineLayout, 0,
                        reduction.m_DescriptorSet, nullptr);
  u32 n_iters = uintlog2(org_size);
  std::vector<u32> strides(n_iters);
  for (u32 i = 0; i < n_iters; i++) {
    strides[i] = 1 << i;
    cb.pushConstants(reduction.m_PipelineLayout,
                     vk::ShaderStageFlagBits::eCompute, 0, 4, &strides[i]);
    cb.dispatch(
        std::clamp(((org_size + 1) / 2) / (64 * strides[i]), 1U, UINT32_MAX), 1,
        1);
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                       vk::PipelineStageFlagBits::eAllCommands, {},
                       FULL_MEMORY_BARRIER, nullptr, nullptr);
  }
}

vk::SurfaceFormatKHR
pick_surface_format(const std::vector<vk::SurfaceFormatKHR>& formats) {
  assert(!formats.empty());
  vk::SurfaceFormatKHR picked_format = formats[0];
  if (formats.size() == 1) {
    if (formats[0].format == vk::Format::eUndefined) {
      picked_format.format = vk::Format::eB8G8R8A8Unorm;
      picked_format.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    }
  } else {
    // request several formats, the first found will be used
    std::array<vk::Format, 4> requested_formats = {
        vk::Format::eB8G8R8A8Unorm, vk::Format::eR8G8B8A8Unorm,
        vk::Format::eB8G8R8Unorm, vk::Format::eR8G8B8Unorm};
    vk::ColorSpaceKHR requested_color_space = vk::ColorSpaceKHR::eSrgbNonlinear;
    for (size_t i = 0;
         i < sizeof(requested_formats) / sizeof(requested_formats[0]); i++) {
      vk::Format requested_format = requested_formats[i];
      auto it =
          std::ranges::find_if(formats.cbegin(), formats.cend(),
                               [requested_format, requested_color_space](
                                   vk::SurfaceFormatKHR const& f) {
                                 return (f.format == requested_format) &&
                                        (f.colorSpace == requested_color_space);
                               });
      if (it != formats.end()) {
        picked_format = *it;
        break;
      }
    }
  }
  assert(picked_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear);
  return picked_format;
}

std::vector<vk::Framebuffer>
create_framebuffers(vk::Device device, std::vector<vk::ImageView> img_views,
                    vk::RenderPass render_pass, vk::Extent2D extent) {
  std::vector<vk::Framebuffer> frame_buffers(img_views.size());
  std::ranges::transform(img_views.begin(), img_views.end(),
                         frame_buffers.begin(), [&](vk::ImageView view) {
                           vk::FramebufferCreateInfo info({}, render_pass, view,
                                                          extent.width,
                                                          extent.height, 1);
                           return device.createFramebuffer(info);
                         });
  return frame_buffers;
}

vk::PresentModeKHR choose_swap_present_mode(
    const std::vector<vk::PresentModeKHR>& present_modes,
    vk::PresentModeKHR requested_mode = vk::PresentModeKHR::eFifo) {
  if (requested_mode == vk::PresentModeKHR::eFifo) {
    return vk::PresentModeKHR::eFifo;
  }
  for (const auto& available_present_mode : present_modes) {
    if (available_present_mode == requested_mode) {
      return available_present_mode;
    }
  }

  std::cout << "Warning: Requested present mode is not available, defaulting "
               "to FIFO.\n";
  return vk::PresentModeKHR::eFifo;
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
  if (qfis[0] != qfis[1]) {
    // If the graphics and present queues are from different queue families,
    // we either have to explicitly transfer ownership of images between the
    // queues, or we have to create the swapchain with imageSharingMode as
    // VK_SHARING_MODE_CONCURRENT
    create_info.imageSharingMode = vk::SharingMode::eConcurrent;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = qfis.data();
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

vk::PhysicalDevice pick_physical_device(const vk::Instance& instance,
                                        const s32 desired_gpu) {
  // check if there are GPUs that support Vulkan and "intelligently" select
  // one. Prioritises discrete GPUs, and after that VRAM size.
  std::vector<vk::PhysicalDevice> p_devices =
      instance.enumeratePhysicalDevices();
  uint32_t n_devices = p_devices.size();

  // shortcut if there's only one device available.
  if (n_devices == 1) {
    if (p_devices[0].getProperties().deviceType ==
            vk::PhysicalDeviceType::eIntegratedGpu or
        p_devices[0].getProperties().deviceType ==
            vk::PhysicalDeviceType::eCpu) {
      spdlog::warn(
          "pick_physical_device: Only integrated GPU or CPU detected, "
          "you may not see much benefit from hardware 'acceleration.'");
    }
    return p_devices[0];
  }

  // Try to select desired GPU if specified.
  if (desired_gpu > -1) {
    if (desired_gpu < static_cast<int32_t>(n_devices)) {
      return p_devices[desired_gpu];
    }
    spdlog::warn("pick_physical_device: Selected device is not available.");
  }

  std::vector<uint32_t> discrete; // the indices of the available discrete gpus
  std::vector<uint64_t> vram(n_devices);
  for (uint32_t i = 0; i < n_devices; i++) {
    if (p_devices[i].getProperties().deviceType ==
        vk::PhysicalDeviceType::eDiscreteGpu) {
      discrete.push_back(i);
    }

    // Gather reported VRAM sizes as an index to rank GPUs by.
    auto heaps = p_devices[i].getMemoryProperties().memoryHeaps;
    for (const auto& heap : heaps) {
      if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
        vram[i] = heap.size;
      }
    }
  }

  // only consider discrete gpus if available:
  if (discrete.size() > 0) {
    if (discrete.size() == 1) {
      return p_devices[discrete[0]];
    }
    const auto max_vram_ptr = std::ranges::max_element(vram);
    return p_devices[std::ranges::distance(vram.begin(), max_vram_ptr)];
  }

  // Otherwise pick integrated with the highest VRAM
  const auto max_vram_ptr = std::ranges::max_element(vram);
  return p_devices[std::ranges::distance(vram.begin(), max_vram_ptr)];
}

Manager::Manager(const AutoInstance& instance, size_t staging_size,
                 std::span<const char*> extra_device_extensions)
    : physical_device(pick_physical_device(*instance)),
      c_qfi(get_compute_queue_family_index(physical_device)) {
  // Validation layers are extremely helpful, we'll only turn them off if we
  // want absolute maximum performance.

  std::vector<const char*> device_extensions{"VK_KHR_maintenance4"};
  for (const auto& ext : extra_device_extensions) {
    device_extensions.push_back(ext);
  }

  // physical_device = pick_physical_device(*instance);

  // c_qfi = get_compute_queue_family_index(physical_device);
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
  // std::vector<vk::SpecializationMapEntry> spec_entries(spec_consts.size());
  // u32 spec_offset = 0;
  // for (u32 i = 0; i < spec_entries.size(); i++) {
  //   spec_entries[i].constantID = i;
  //   spec_entries[i].offset = spec_offset;
  //   spec_entries[i].size = 4;
  //   spec_offset += 4;
  // }
  // vk::SpecializationInfo spec_info;
  // spec_info.mapEntryCount = spec_consts.size();
  // spec_info.pMapEntries = spec_entries.data();
  // spec_info.dataSize = spec_offset;
  // spec_info.pData = spec_consts.data();
  // const auto spec_info = span_to_spec_consts(spec_consts);
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

Renderer::Renderer(Manager& manager, vk::SurfaceKHR surf, u32 nx, u32 ny)
    : p_mgr{&manager}, surface{surf} {
  render_queue_indices = {manager.g_qfi, manager.p_qfi};
  if (render_queue_indices[0] != render_queue_indices[1]) {
    runtime_exc("Graphics queue is different from present queue, I don't want "
                "to deal with that right now!");
  }
  graphics_queue = manager.device.getQueue(render_queue_indices[0], 0);
  present_queue = manager.device.getQueue(render_queue_indices[1], 0);
  capabilities = manager.physical_device.getSurfaceCapabilitiesKHR(surface);
  auto formats = manager.physical_device.getSurfaceFormatsKHR(surface);
  if (formats.empty()) {
    runtime_exc("Fatal: No surface formats found");
  }
  surface_format = pick_surface_format(formats);
  auto present_modes =
      manager.physical_device.getSurfacePresentModesKHR(surface);
  if (present_modes.empty()) {
    runtime_exc("Fatal: No present modes found");
  }
  vk::CommandPoolCreateInfo command_pool_ci(vk::CommandPoolCreateFlags(),
                                            render_queue_indices[0]);
  command_pool = manager.device.createCommandPool(command_pool_ci);

  present_mode = choose_swap_present_mode(present_modes);
  if (capabilities.currentExtent.width ==
      std::numeric_limits<uint32_t>::max()) {
    s32 width = 0;
    s32 height = 0;
    u32 uwidth = (u32)width;
    u32 uheight = (u32)height;
    swapchain_extent.width =
        std::clamp(uwidth, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.height);
    swapchain_extent.height =
        std::clamp(uheight, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

  } else {
    swapchain_extent.width = std::clamp(capabilities.currentExtent.width,
                                        capabilities.minImageExtent.width,
                                        capabilities.maxImageExtent.width);
    swapchain_extent.height = std::clamp(capabilities.currentExtent.height,
                                         capabilities.minImageExtent.height,
                                         capabilities.maxImageExtent.height);
  }

  n_images = capabilities.maxImageCount == 0
                 ? 3U
                 : std::clamp(3U, capabilities.minImageCount,
                              capabilities.minImageCount);
  vk::SurfaceTransformFlagBitsKHR pre_transform =
      (capabilities.supportedTransforms &
       vk::SurfaceTransformFlagBitsKHR::eIdentity)
          ? vk::SurfaceTransformFlagBitsKHR::eIdentity
          : capabilities.currentTransform;
  capabilities.currentTransform = pre_transform;

  swapchain = create_swapchain(manager.device, surface, capabilities,
                               surface_format, present_mode, swapchain_extent,
                               n_images, render_queue_indices);

  swapchain_imgs = manager.device.getSwapchainImagesKHR(swapchain);
  swapchain_img_views.reserve(swapchain_imgs.size());
  vk::ImageViewCreateInfo img_view_create_info(
      {}, {}, vk::ImageViewType::e2D, surface_format.format, {},
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  for (auto img : swapchain_imgs) {
    img_view_create_info.image = img;
    swapchain_img_views.push_back(
        manager.device.createImageView(img_view_create_info));
  }

  vk::AttachmentDescription color_attachment(
      vk::AttachmentDescriptionFlags(), surface_format.format,
      vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
      vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
      vk::ImageLayout::ePresentSrcKHR);

  vk::AttachmentReference color_attachment_ref(
      0, vk::ImageLayout::eColorAttachmentOptimal);

  vk::SubpassDescription subpass(vk::SubpassDescriptionFlags(),
                                 vk::PipelineBindPoint::eGraphics, {},
                                 color_attachment_ref);

  vk::SubpassDependency dependency{};
  dependency.setSrcSubpass(vk::SubpassExternal);
  dependency.setDstSubpass(0);
  dependency.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  dependency.setSrcAccessMask(vk::AccessFlagBits::eNone);
  dependency.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  dependency.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                              vk::AccessFlagBits::eColorAttachmentWrite);

  vk::RenderPassCreateInfo render_pass_info({}, color_attachment, subpass,
                                            dependency);

  render_pass = manager.device.createRenderPass(render_pass_info);
  swapchain_fbs = create_framebuffers(manager.device, swapchain_img_views,
                                      render_pass, swapchain_extent);

  // std::string base_path = "build/";
  auto vert_code = read_file<u32>("Shaders/quad.vert.spv");
  auto frag_code = read_file<u32>("Shaders/quad.frag.spv");

  vk::ShaderModuleCreateInfo vert_mci(vk::ShaderModuleCreateFlags(), vert_code);
  vk::ShaderModule vert_module = manager.device.createShaderModule(vert_mci);
  vk::ShaderModuleCreateInfo frag_mci(vk::ShaderModuleCreateFlags(), frag_code);
  vk::ShaderModule frag_module = manager.device.createShaderModule(frag_mci);
  if (vert_module == VK_NULL_HANDLE) {
    runtime_exc("Failed to create vertex module");
  }
  if (frag_module == VK_NULL_HANDLE) {
    runtime_exc("Failed to create frag module");
  }

  vk::PipelineShaderStageCreateInfo vert_stage_info(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
      vert_module, "main");

  vk::PipelineShaderStageCreateInfo frag_stage_info(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment,
      frag_module, "main");
  std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages = {
      vert_stage_info, frag_stage_info};

  auto vertex_binding = PositionTextureVertex::binding_dscr();
  auto attr = PositionTextureVertex::attribute_dscr();
  vk::PipelineVertexInputStateCreateInfo vertex_input_info(
      vk::PipelineVertexInputStateCreateFlags(), vertex_binding, attr);

  vk::PipelineInputAssemblyStateCreateInfo input_assembly(
      vk::PipelineInputAssemblyStateCreateFlags(),
      vk::PrimitiveTopology::eTriangleList);
  vk::Viewport viewport(0.0, 0.0, (f32)swapchain_extent.width,
                        (f32)swapchain_extent.height, 0.0F, 1.0F);

  vk::Rect2D scissor(vk::Offset2D(0, 0), swapchain_extent);

  vk::PipelineViewportStateCreateInfo viewport_state(
      vk::PipelineViewportStateCreateFlags(), 1, &viewport, 1, &scissor);
  vk::PipelineRasterizationStateCreateInfo rasterizer(
      vk::PipelineRasterizationStateCreateFlags(),
      vk::False,                   // depthClampEnable
      vk::False,                   // rasterizerDiscardEnable
      vk::PolygonMode::eFill,      // polygonMode
      vk::CullModeFlagBits::eBack, // cullMode
      vk::FrontFace::eClockwise,   // frontFace
      vk::False,                   // depthBiasEnable
      0.0F,                        // depthBiasConstantFactor
      0.0F,                        // depthBiasClamp
      0.0F,                        // depthBiasSlopeFactor
      1.0F                         // lineWidth
  );

  vk::PipelineMultisampleStateCreateInfo multisampling(
      vk::PipelineMultisampleStateCreateFlags(), // flags
      vk::SampleCountFlagBits::e1                // rasterizationSamples
                                                 // other values can be default
  );

  vk::ColorComponentFlags color_component_flags(
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  vk::PipelineColorBlendAttachmentState color_blend_attachment(
      vk::False,              // blendEnable
      vk::BlendFactor::eZero, // srcColorBlendFactor
      vk::BlendFactor::eZero, // dstColorBlendFactor
      vk::BlendOp::eAdd,      // colorBlendOp
      vk::BlendFactor::eZero, // srcAlphaBlendFactor
      vk::BlendFactor::eZero, // dstAlphaBlendFactor
      vk::BlendOp::eAdd,      // alphaBlendOp
      color_component_flags   // colorWriteMask
  );

  vk::PipelineColorBlendStateCreateInfo color_blending(
      vk::PipelineColorBlendStateCreateFlags(), // flags
      vk::False,                                // logicOpEnable
      vk::LogicOp::eCopy,                       // logicOp
      color_blend_attachment,                   // attachments
      {{1.0F, 1.0F, 1.0F, 1.0F}}                // blendConstants
  );

  vk::DescriptorSetLayoutBinding dsl_binding(
      0, vk::DescriptorType::eCombinedImageSampler, 1,
      vk::ShaderStageFlagBits::eFragment);
  descriptor_set_layout = manager.device.createDescriptorSetLayout(
      vk::DescriptorSetLayoutCreateInfo({}, dsl_binding));
  vk::PipelineLayoutCreateInfo graphics_pipeline_layout_info{};
  graphics_pipeline_layout_info.setSetLayouts(descriptor_set_layout);

  graphics_pipeline_layout =
      manager.device.createPipelineLayout(graphics_pipeline_layout_info);

  std::vector<vk::DynamicState> dynamic_states = {vk::DynamicState::eViewport,
                                                  vk::DynamicState::eScissor};

  vk::PipelineDynamicStateCreateInfo dynamic_info(
      vk::PipelineDynamicStateCreateFlags(), dynamic_states);

  vk::GraphicsPipelineCreateInfo pipeline_info(
      vk::PipelineCreateFlags(), shader_stages, &vertex_input_info,
      &input_assembly, nullptr, &viewport_state, &rasterizer, &multisampling,
      nullptr, &color_blending, &dynamic_info, graphics_pipeline_layout,
      render_pass);

  auto result = manager.device.createGraphicsPipeline({}, pipeline_info);
  assert(result.result == vk::Result::eSuccess);
  graphics_pipeline = result.value;
  manager.device.destroyShaderModule(frag_module, nullptr);
  manager.device.destroyShaderModule(vert_module, nullptr);

  std::vector<PositionTextureVertex> vertices = {
      {.pos = {-1.0F, -1.0F}, .uv = {0.0F, 0.0F}},
      {.pos = {1.0F, -1.0F}, .uv = {1.0F, 0.0F}},
      {.pos = {1.0F, 1.0F}, .uv = {1.0F, 1.0F}},
      {.pos = {1.0F, 1.0F}, .uv = {1.0F, 1.0F}},
      {.pos = {-1.0F, 1.0F}, .uv = {0.0F, 1.0F}},
      {.pos = {-1.0F, -1.0F}, .uv = {0.0F, 0.0F}},
      {.pos = {1.0F, -1.0F}, .uv = {0.0F, 0.0F}},
      {.pos = {1.0F, -1.0F}, .uv = {0.0F, 0.0F}},
      {.pos = {1.0F, 1.0F}, .uv = {0.0F, 1.0F}},
      {.pos = {1.0F, -1.0F}, .uv = {0.0F, 0.0F}},
      {.pos = {1.0F, 1.0F}, .uv = {0.0F, 1.0F}},
      {.pos = {1.0F, -1.0F}, .uv = {0.0F, 1.0F}}};
  vk::BufferCreateInfo vertex_bci(
      {}, vertices.size() * sizeof(PositionTextureVertex),
      vk::BufferUsageFlagBits::eVertexBuffer |
          vk::BufferUsageFlagBits::eTransferDst);
  VmaAllocationCreateInfo alloc_create_info{};
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
  alloc_create_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  alloc_create_info.priority = 1.0F;
  vertex_buffer.allocate(manager.allocator, alloc_create_info, vertex_bci);
  manager.write_to_buffer(vertex_buffer, vertices);

  vk::Format colormap_format = vk::Format::eR32G32B32A32Sfloat;
  vk::ImageCreateInfo colormap_img_info(
      {}, vk::ImageType::e2D, colormap_format, vk::Extent3D(nx, ny, 1), 1, 1,
      vk::SampleCountFlagBits::e1, vk::ImageTiling::eLinear,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);
  VmaAllocationCreateInfo img_alloc_create_info{};
  img_alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
  img_alloc_create_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  alloc_create_info.priority = 1.0F;
  colormap_img.allocate(manager.allocator, img_alloc_create_info,
                        colormap_img_info);
  vk::ImageMemoryBarrier initialbarrier{};
  initialbarrier.setOldLayout(vk::ImageLayout::eUndefined);
  // On Nvidia there is really only the General image layout. On AMD General
  // is compressed so you need to switch to a different layout for adequate
  // performance there
  initialbarrier.setNewLayout(vk::ImageLayout::eGeneral);
  initialbarrier.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
  initialbarrier.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
  initialbarrier.setImage(colormap_img.img);
  initialbarrier.setSubresourceRange(
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  initialbarrier.setSrcAccessMask({});
  initialbarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  one_time_submit(manager.device, manager.command_pool, manager.queue,
                  [&](vk::CommandBuffer b) {
                    b.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                      vk::PipelineStageFlagBits::eComputeShader,
                                      {}, nullptr, nullptr, initialbarrier);
                  });

  vk::ImageViewCreateInfo view_info(
      {}, colormap_img.img, vk::ImageViewType::e2D, colormap_format, {},
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  colormap_view = manager.device.createImageView(view_info);

  vk::BufferCreateInfo colormap_bci({}, 256 * sizeof(cm::AlignedColor),
                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                        vk::BufferUsageFlagBits::eTransferDst |
                                        vk::BufferUsageFlagBits::eTransferSrc);
  colormap.allocate(manager.allocator, img_alloc_create_info, colormap_bci);
  manager.write_to_buffer(colormap, cm::viridis.data(),
                          cm::viridis.size() * sizeof(cm::AlignedColor));
  const u32 source_length = nx * ny;
  vk::BufferCreateInfo value_bci({}, source_length * sizeof(f32),
                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                     vk::BufferUsageFlagBits::eTransferDst |
                                     vk::BufferUsageFlagBits::eTransferSrc);
  const u32 n_target = (ny * nx) / 16;
  vk::BufferCreateInfo minmax_bci({}, n_target * sizeof(f32),
                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                      vk::BufferUsageFlagBits::eTransferDst |
                                      vk::BufferUsageFlagBits::eTransferSrc);
  value_buffer.allocate(manager.allocator, img_alloc_create_info, value_bci);
  minmax_buffer.allocate(manager.allocator, img_alloc_create_info, minmax_bci);
  std::vector<f32> values(source_length);
  for (u32 j = 0; j < ny; j++) {
    for (u32 i = 0; i < nx; i++) {
      values[nx * j + i] = (f32)(i + j);
    }
  }
  manager.write_to_buffer(value_buffer, values);

  const auto first_minmax_code = read_file<u32>("Shaders/firstminmax.spv");
  auto minmax_code = read_file<u32>("Shaders/minmax.spv");
  // const std::vector<f32> first_minmax_specs{bit_cast<f32>(source_length),
  //                                           bit_cast<f32>(n_target)};
  std::span<const f32> first_minmax_specs =
      spec_span(&source_length, &source_length + 1);
  std::span<const f32> minmax_specs = spec_span(&n_target, &n_target + 1);
  spdlog::debug("Initializing first minmax reduction algorithm");
  first_minmax_reduction.initialize(manager.device, first_minmax_code, 0, 2, 0,
                                    first_minmax_specs);
  std::vector<const MetaBuffer*> first_minmax_buffers{&value_buffer,
                                                      &minmax_buffer};
  first_minmax_reduction.bind_data({}, first_minmax_buffers, {});
  spdlog::debug("Initializing minmax reduction algorithm");
  std::vector<const MetaBuffer*> minmax_buffers{&minmax_buffer};
  minmax_reduction.initialize(manager.device, minmax_code, 0, 1, 0,
                              minmax_specs);
  minmax_reduction.bind_data({}, minmax_buffers, {});

  auto fill_code = read_file<u32>("Shaders/colormap.spv");
  spdlog::debug("Initializing colormap algorithm");
  fill_colormap_img.initialize(manager.device, fill_code, 1, 3, 0,
                               minmax_specs);
  std::vector<const MetaBuffer*> colormap_buffers{&colormap, &value_buffer,
                                                  &minmax_buffer};
  fill_colormap_img.bind_data({&colormap_view, &colormap_view + 1},
                              colormap_buffers, {});

  vk::SamplerCreateInfo colormap_sampler_info(
      vk::SamplerCreateFlags(), vk::Filter::eNearest, vk::Filter::eNearest,
      vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge, 0.0F, vk::False, 1.0F, vk::False,
      vk::CompareOp::eNever, 0.0F, 0.0F, vk::BorderColor::eFloatOpaqueWhite);
  colormap_sampler = manager.device.createSampler(colormap_sampler_info);

  vk::DescriptorPoolSize pool_size(vk::DescriptorType::eCombinedImageSampler,
                                   MAX_FRAMES_IN_FLIGHT);
  descriptor_pool = manager.device.createDescriptorPool(
      {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, n_images, 1,
       &pool_size});

  descriptor_set =
      manager.device
          .allocateDescriptorSets({descriptor_pool, descriptor_set_layout})
          .front();
  vk::DescriptorImageInfo descriptor_img_info(
      colormap_sampler, colormap_view, vk::ImageLayout::eShaderReadOnlyOptimal);
  vk::WriteDescriptorSet descriptor_write(
      descriptor_set, 0, 0, vk::DescriptorType::eCombinedImageSampler,
      descriptor_img_info);
  manager.device.updateDescriptorSets(descriptor_write, {});

  image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
  image_in_flight_fences.resize(swapchain_imgs.size(), VK_NULL_HANDLE);

  vk::SemaphoreCreateInfo semaphore_info{};

  vk::FenceCreateInfo fence_info(vk::FenceCreateFlagBits::eSignaled);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    image_available_semaphores[i] =
        manager.device.createSemaphore(semaphore_info);
    render_finished_semaphores[i] =
        manager.device.createSemaphore(semaphore_info);
    in_flight_fences[i] = manager.device.createFence(fence_info);
  }

  vk::CommandBufferAllocateInfo cb_alloc_info(
      command_pool, vk::CommandBufferLevel::ePrimary, n_images);
  command_buffers.resize(n_images);
  vkAllocateCommandBuffers(p_mgr->device,
                           pcast<VkCommandBufferAllocateInfo>(&cb_alloc_info),
                           pcast<VkCommandBuffer>(command_buffers.data()));
  vk::CommandBufferBeginInfo begin_info{};
  reduction_buffer = manager.begin_record();
  vk::ImageMemoryBarrier present_to_storage{};
  present_to_storage.setNewLayout(vk::ImageLayout::eGeneral);
  present_to_storage.setOldLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
  present_to_storage.setSrcQueueFamilyIndex(manager.c_qfi);
  present_to_storage.setDstQueueFamilyIndex(manager.c_qfi);
  present_to_storage.setDstAccessMask(vk::AccessFlagBits::eShaderWrite);
  present_to_storage.setSrcAccessMask(vk::AccessFlagBits::eShaderRead);
  present_to_storage.setImage(colormap_img.img);
  present_to_storage.setSubresourceRange(
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  reduction_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   vk::PipelineStageFlagBits::eComputeShader,
                                   {}, nullptr, nullptr, present_to_storage);
  u32 x = (nx * ny) / 32;
  append_op(reduction_buffer, first_minmax_reduction, x, 1, 1);
  x = (x + 31) / 32;
  while (x > 1) {
    append_op(reduction_buffer, minmax_reduction, x, 1, 1);
    x = (x + 31) / 32;
  }
  append_op(reduction_buffer, minmax_reduction, 1, 1, 1);
  append_op(reduction_buffer, fill_colormap_img, nx / 8, ny / 4, 1);
  vk::ImageMemoryBarrier storage_to_present{};
  storage_to_present.setOldLayout(vk::ImageLayout::eGeneral);
  storage_to_present.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
  storage_to_present.setSrcQueueFamilyIndex(manager.c_qfi);
  storage_to_present.setDstQueueFamilyIndex(manager.c_qfi);
  storage_to_present.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  storage_to_present.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
  storage_to_present.setImage(colormap_img.img);
  storage_to_present.setSubresourceRange(
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  reduction_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   vk::PipelineStageFlagBits::eComputeShader,
                                   {}, nullptr, nullptr, storage_to_present);
  reduction_buffer.end();
  one_time_submit(manager.device, manager.command_pool, manager.queue,
                  [&](vk::CommandBuffer b) {
                    b.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                      vk::PipelineStageFlagBits::eComputeShader,
                                      {}, nullptr, nullptr, storage_to_present);
                  });
  for (size_t i = 0; i < n_images; i++) {
    command_buffers[i].begin(begin_info);
    record_drawing_commands(swapchain_fbs[i], render_pass, swapchain_extent,
                            graphics_pipeline, graphics_pipeline_layout,
                            descriptor_set, vertex_buffer.buffer,
                            command_buffers[i]);
    command_buffers[i].end();
  }
}

void Renderer::cleanup_swapchain() {
  vk::Device dev = p_mgr->device;
  dev.destroyCommandPool(command_pool);
  for (auto& fb : swapchain_fbs) {
    dev.destroyFramebuffer(fb);
  }
  for (auto& iv : swapchain_img_views) {
    dev.destroyImageView(iv);
  }
  dev.destroySwapchainKHR(swapchain);
}
void Renderer::recreate_swapchain() {
  p_mgr->device.waitIdle();
  capabilities = p_mgr->physical_device.getSurfaceCapabilitiesKHR(surface);
  if (capabilities.currentExtent.width ==
      std::numeric_limits<uint32_t>::max()) {
    s32 width = 0;
    s32 height = 0;
    // GetWindowSize seems to just return 0 on linux for this version of SDL3,
    // hopefully we never encounter this branch.
    u32 uwidth = (u32)width;
    u32 uheight = (u32)height;
    swapchain_extent.width =
        std::clamp(uwidth, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.height);
    swapchain_extent.height =
        std::clamp(uheight, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

  } else {
    swapchain_extent.width = std::clamp(capabilities.currentExtent.width,
                                        capabilities.minImageExtent.width,
                                        capabilities.maxImageExtent.width);
    swapchain_extent.height = std::clamp(capabilities.currentExtent.height,
                                         capabilities.minImageExtent.height,
                                         capabilities.maxImageExtent.height);
  }
  vk::SwapchainKHR new_swapchain = create_swapchain(
      p_mgr->device, surface, capabilities, surface_format, present_mode,
      swapchain_extent, n_images, render_queue_indices, swapchain);
  cleanup_swapchain();
  swapchain = new_swapchain;
  swapchain_imgs = p_mgr->device.getSwapchainImagesKHR(swapchain);
  swapchain_img_views =
      create_image_views(p_mgr->device, surface_format, swapchain_imgs);

  vk::CommandPoolCreateInfo command_pool_info({}, render_queue_indices[0]);
  swapchain_fbs = create_framebuffers(p_mgr->device, swapchain_img_views,
                                      render_pass, swapchain_extent);
  command_pool = p_mgr->device.createCommandPool(command_pool_info);
  vk::CommandBufferAllocateInfo cb_alloc_info(
      command_pool, vk::CommandBufferLevel::ePrimary, n_images);
  command_buffers.resize(n_images);
  vkAllocateCommandBuffers(p_mgr->device,
                           pcast<VkCommandBufferAllocateInfo>(&cb_alloc_info),
                           pcast<VkCommandBuffer>(command_buffers.data()));
  vk::CommandBufferBeginInfo begin_info{};
  for (u32 i = 0; i < n_images; i++) {
    command_buffers[i].begin(begin_info);
    record_drawing_commands(swapchain_fbs[i], render_pass, swapchain_extent,
                            graphics_pipeline, graphics_pipeline_layout,
                            descriptor_set, vertex_buffer.buffer,
                            command_buffers[i]);
    command_buffers[i].end();
  }
}

void Renderer::draw_frame() {
  vk::Device dev = p_mgr->device;
  if (dev.waitForFences(in_flight_fences[current_frame], vk::True, -1) !=
      vk::Result::eSuccess) {
    throw runtime_exc("Failed waiting for fences");
  };
  try {
    vk::ResultValue<u32> res = dev.acquireNextImageKHR(
        swapchain, UINT64_MAX, image_available_semaphores[current_frame],
        nullptr);
    if (res.result != vk::Result::eSuccess &&
        res.result != vk::Result::eSuboptimalKHR) {
      throw runtime_exc("Failed to acquire swap chain image!");
    }
    u32 img_index = res.value;
    if (image_in_flight_fences[img_index] != nullptr) {
      auto result =
          dev.waitForFences(image_in_flight_fences[img_index], vk::True, -1);
      if (result != vk::Result::eSuccess) {
        throw runtime_exc("Failed to wait on in flight image");
      }
    }
    image_in_flight_fences[img_index] = in_flight_fences[current_frame];

    vk::Semaphore signal_semaphore = render_finished_semaphores[current_frame];
    vk::PipelineStageFlags wait_dst_stage_mask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::CommandBuffer cb = command_buffers[img_index];
    vk::SubmitInfo submit_info(image_available_semaphores[current_frame],
                               wait_dst_stage_mask, cb, signal_semaphore);
    dev.resetFences(in_flight_fences[current_frame]);
    p_mgr->execute(reduction_buffer);

    graphics_queue.submit(submit_info, in_flight_fences[current_frame]);

    vk::PresentInfoKHR pres_info(signal_semaphore, swapchain, img_index);
    try {
      if (vk::Result::eSuboptimalKHR == present_queue.presentKHR(pres_info)) {
        std::cout << "Recreating swapchain because it is suboptimal\n";
        recreate_swapchain();
      }
    } catch (vk::OutOfDateKHRError& out_of_date_error) {
      recreate_swapchain();
      return;
    }

    current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
  } catch (vk::OutOfDateKHRError& error) {
    recreate_swapchain();
    return;
  };
}

Renderer::~Renderer() {
  vk::Device dev = p_mgr->device;
  dev.waitIdle();
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    dev.destroySemaphore(image_available_semaphores[i]);
    dev.destroySemaphore(render_finished_semaphores[i]);
    dev.destroyFence(in_flight_fences[i]);
  }

  for (size_t i = 0; i < n_images; i++) {
    dev.destroyFramebuffer(swapchain_fbs[i]);
    dev.destroyImageView(swapchain_img_views[i]);
  }
  dev.destroyCommandPool(command_pool);
  dev.destroyDescriptorPool(descriptor_pool);
  dev.destroyDescriptorSetLayout(descriptor_set_layout);
  dev.destroySampler(colormap_sampler);
  dev.destroyImageView(colormap_view);
  dev.destroySwapchainKHR(swapchain);
  dev.destroyPipeline(graphics_pipeline);
  dev.destroyPipelineLayout(graphics_pipeline_layout);
  dev.destroyRenderPass(render_pass);
}
