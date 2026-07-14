#include "Hal/Gpu/Internal/VulkanUtil.h"

#include "Hal/Log/Logger.h"

#include "Hal/Gpu/Internal/VmaUsage.h"

#include <fstream>
#include <stdexcept>
#include <string>

namespace Hal
{
    namespace Gpu
    {
        namespace Internal
        {
            void CheckResult(VkResult    result,
                             const char *pMessage)
            {
                if (result != VK_SUCCESS)
                {
                    std::string fullMessage = std::string(pMessage) + " (VkResult " + std::to_string(result) + ")";
                    Logger::Instance().Log(LogSeverity::Error, fullMessage);
                    throw std::runtime_error(fullMessage);
                }
            }

            uint32_t FindMemoryType(VkPhysicalDevice       physicalDevice,
                                    uint32_t               typeFilter,
                                    VkMemoryPropertyFlags  properties)
            {
                VkPhysicalDeviceMemoryProperties memProperties;
                vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
                for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
                {
                    bool typeMatches = (typeFilter & (1u << i)) != 0;
                    bool propsMatch = (memProperties.memoryTypes[i].propertyFlags & properties) == properties;
                    if (typeMatches && propsMatch)
                    {
                        return i;
                    }
                }
                Logger::Instance().Log(LogSeverity::Error, "no suitable Vulkan memory type found");
                throw std::runtime_error("no suitable Vulkan memory type found");
            }

            VkBuffer CreateBuffer(VkPhysicalDevice       physicalDevice,
                                  VkDevice               device,
                                  VkDeviceSize           size,
                                  VkBufferUsageFlags     usage,
                                  VkMemoryPropertyFlags  properties,
                                  VkDeviceMemory        &outMemory)
            {
                VkBufferCreateInfo bufferInfo{};
                bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bufferInfo.size = size;
                bufferInfo.usage = usage;
                bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                VkBuffer buffer = VK_NULL_HANDLE;
                CheckResult(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer), "vkCreateBuffer failed");

                VkMemoryRequirements memRequirements;
                vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                allocInfo.allocationSize = memRequirements.size;
                allocInfo.memoryTypeIndex = FindMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

                CheckResult(vkAllocateMemory(device, &allocInfo, nullptr, &outMemory), "vkAllocateMemory failed");
                CheckResult(vkBindBufferMemory(device, buffer, outMemory, 0), "vkBindBufferMemory failed");
                return buffer;
            }

            std::vector<char> ReadFile(const std::string &path)
            {
                std::ifstream file(path, std::ios::ate | std::ios::binary);
                if (!file.is_open())
                {
                    std::string message = "failed to open shader file: " + path;
                    Logger::Instance().Log(LogSeverity::Error, message);
                    throw std::runtime_error(message);
                }
                size_t fileSize = static_cast<size_t>(file.tellg());
                std::vector<char> buffer(fileSize);
                file.seekg(0);
                file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
                return buffer;
            }

            bool HasInstanceExtension(const std::vector<VkExtensionProperties> &available,
                                      const char                               *pName)
            {
                for (const VkExtensionProperties &ext : available)
                {
                    if (std::string(ext.extensionName) == pName)
                    {
                        return true;
                    }
                }
                return false;
            }

            bool HasDeviceExtension(const std::vector<VkExtensionProperties> &available,
                                    const char                               *pName)
            {
                for (const VkExtensionProperties &ext : available)
                {
                    if (std::string(ext.extensionName) == pName)
                    {
                        return true;
                    }
                }
                return false;
            }

            VkBuffer CreateDeviceBufferVma(VmaAllocator        allocator,
                                           VkDeviceSize        size,
                                           VkBufferUsageFlags  usage,
                                           VmaAllocation      &outAllocation)
            {
                VkBufferCreateInfo bufferInfo = {};
                bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bufferInfo.size = size;
                bufferInfo.usage = usage;
                bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                VmaAllocationCreateInfo allocInfo = {};
                allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

                VkBuffer buffer = VK_NULL_HANDLE;
                CheckResult(vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &outAllocation, nullptr), "vmaCreateBuffer (device) failed");
                return buffer;
            }

            VkBuffer CreateStagingBufferVma(VmaAllocator        allocator,
                                            VkDeviceSize        size,
                                            VkBufferUsageFlags  usage,
                                            VmaAllocation      &outAllocation,
                                            void              **ppMapped)
            {
                VkBufferCreateInfo bufferInfo = {};
                bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bufferInfo.size = size;
                bufferInfo.usage = usage;
                bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                // HOST_ACCESS_RANDOM (not SEQUENTIAL) because readback
                // staging is read by the host; VMA guarantees a
                // HOST_VISIBLE|HOST_COHERENT type for it. CREATE_MAPPED
                // keeps it mapped for the buffer's whole lifetime.
                VmaAllocationCreateInfo allocInfo = {};
                allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
                allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

                VkBuffer buffer = VK_NULL_HANDLE;
                VmaAllocationInfo resultInfo = {};
                CheckResult(vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &outAllocation, &resultInfo), "vmaCreateBuffer (staging) failed");
                *ppMapped = resultInfo.pMappedData;
                return buffer;
            }

            void DestroyBufferVma(VmaAllocator  allocator,
                                  VkBuffer      buffer,
                                  VmaAllocation allocation)
            {
                if (buffer != VK_NULL_HANDLE)
                {
                    vmaDestroyBuffer(allocator, buffer, allocation);
                }
            }
        }
    }
}
