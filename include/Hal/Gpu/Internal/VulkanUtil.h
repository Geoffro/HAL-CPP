#pragma once

#include <volk.h>

#include <string>
#include <vector>

// Forward declarations of VMA's handles (declared identically in
// vk_mem_alloc.h), so this header doesn't pull the whole VMA header in.
VK_DEFINE_HANDLE(VmaAllocator)
VK_DEFINE_HANDLE(VmaAllocation)

namespace Hal
{
    namespace Gpu
    {
        namespace Internal
        {
            // Throws std::runtime_error (after logging via Hal::Logger) if
            // result != VK_SUCCESS. Shared by every Hal::Gpu translation
            // unit so Vulkan call sites stay a single line.
            void CheckResult(VkResult    result,
                             const char *pMessage);

            // Finds a memory type index on physicalDevice satisfying both
            // typeFilter (a bitmask, as returned by
            // vkGetBufferMemoryRequirements/vkGetImageMemoryRequirements)
            // and properties. Shared by every Hal::Gpu context/kernel that
            // allocates its own buffers or images — takes plain handles
            // rather than a Hal::Gpu::ComputeContext so it's equally usable
            // from WindowRenderContext, which owns a separate device.
            uint32_t FindMemoryType(VkPhysicalDevice       physicalDevice,
                                    uint32_t               typeFilter,
                                    VkMemoryPropertyFlags  properties);

            // Creates a buffer and a bound, dedicated memory allocation for
            // it. Still used by the window render contexts, which own
            // devices without a VMA allocator; the compute kernels allocate
            // through the VMA helpers below instead. outMemory receives the
            // allocation, which the caller owns and must free.
            VkBuffer CreateBuffer(VkPhysicalDevice       physicalDevice,
                                  VkDevice               device,
                                  VkDeviceSize           size,
                                  VkBufferUsageFlags     usage,
                                  VkMemoryPropertyFlags  properties,
                                  VkDeviceMemory        &outMemory);

            // VMA-allocated device-local buffer (VMA picks the memory
            // type, preferring device-local; no host access). outAllocation
            // is freed with DestroyBufferVma.
            VkBuffer CreateDeviceBufferVma(VmaAllocator        allocator,
                                           VkDeviceSize        size,
                                           VkBufferUsageFlags  usage,
                                           VmaAllocation      &outAllocation);

            // VMA-allocated, persistently mapped host staging buffer:
            // ppMapped receives the mapping, valid for the buffer's whole
            // lifetime (no per-use vkMapMemory/vkUnmapMemory), host-
            // coherent. Readable and writable from the host (upload and
            // readback staging share one helper).
            VkBuffer CreateStagingBufferVma(VmaAllocator        allocator,
                                            VkDeviceSize        size,
                                            VkBufferUsageFlags  usage,
                                            VmaAllocation      &outAllocation,
                                            void              **ppMapped);

            void DestroyBufferVma(VmaAllocator  allocator,
                                  VkBuffer      buffer,
                                  VmaAllocation allocation);

            // Reads an entire binary file (a compiled .spv shader) into
            // memory. Shared by every Hal::Gpu kernel that loads a shader
            // module.
            std::vector<char> ReadFile(const std::string &path);

            // True if pName appears in available (as returned by
            // vkEnumerateInstance/DeviceExtensionProperties). Shared by
            // every Hal::Gpu context that opportunistically enables an
            // extension only if the instance/device actually advertises
            // it (e.g. MoltenVK's portability extensions).
            bool HasInstanceExtension(const std::vector<VkExtensionProperties> &available,
                                      const char                               *pName);
            bool HasDeviceExtension(const std::vector<VkExtensionProperties> &available,
                                    const char                               *pName);
        }
    }
}
