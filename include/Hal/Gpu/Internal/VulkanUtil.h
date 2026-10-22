#pragma once

#include <volk.h>

#include <cstddef>
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
            // Throws std::runtime_error (after logging) if
            // result != VK_SUCCESS.
            void CheckResult(VkResult    result,
                             const char *pMessage);

            // Throws std::invalid_argument (after logging pMessage) unless
            // condition holds.
            void CheckArgument(bool        condition,
                               const char *pMessage);

            // Throws std::invalid_argument (after logging) unless
            // actual == expected. pWhat names the call and argument.
            void CheckSize(size_t      actual,
                           size_t      expected,
                           const char *pWhat);

            // Finds a memory type index satisfying both typeFilter (from
            // vkGet*MemoryRequirements) and properties.
            uint32_t FindMemoryType(VkPhysicalDevice       physicalDevice,
                                    uint32_t               typeFilter,
                                    VkMemoryPropertyFlags  properties);

            // Creates a buffer with its own bound memory allocation (for
            // devices without a VMA allocator). The caller owns outMemory.
            VkBuffer CreateBuffer(VkPhysicalDevice       physicalDevice,
                                  VkDevice               device,
                                  VkDeviceSize           size,
                                  VkBufferUsageFlags     usage,
                                  VkMemoryPropertyFlags  properties,
                                  VkDeviceMemory        &outMemory);

            // VMA-allocated buffer preferring device-local memory, freed
            // with DestroyBufferVma.
            VkBuffer CreateDeviceBufferVma(VmaAllocator        allocator,
                                           VkDeviceSize        size,
                                           VkBufferUsageFlags  usage,
                                           VmaAllocation      &outAllocation);

            // VMA-allocated host-coherent staging buffer, readable and
            // writable from the host and mapped for its whole lifetime
            // (ppMapped receives the mapping).
            VkBuffer CreateStagingBufferVma(VmaAllocator        allocator,
                                            VkDeviceSize        size,
                                            VkBufferUsageFlags  usage,
                                            VmaAllocation      &outAllocation,
                                            void              **ppMapped);

            void DestroyBufferVma(VmaAllocator  allocator,
                                  VkBuffer      buffer,
                                  VmaAllocation allocation);

            // Reads an entire binary file (a compiled .spv shader).
            std::vector<char> ReadFile(const std::string &path);

            // True if pName appears in available (from
            // vkEnumerateInstance/DeviceExtensionProperties).
            bool HasExtension(const std::vector<VkExtensionProperties> &available,
                              const char                               *pName);
        }
    }
}
