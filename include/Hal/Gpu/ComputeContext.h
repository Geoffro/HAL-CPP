#pragma once

#include <volk.h>

#include <cstdint>

// VMA's handle, forward-declared so this header doesn't include
// vk_mem_alloc.h.
VK_DEFINE_HANDLE(VmaAllocator)

namespace Hal
{
    namespace Gpu
    {
        // Headless Vulkan compute context: instance, physical device, logical
        // device, compute queue, and the VMA allocator the kernels use. Kernels
        // must be destroyed before their context.
        class ComputeContext
        {
        public:
            ComputeContext();
            ~ComputeContext();

            ComputeContext(const ComputeContext &other) = delete;
            ComputeContext &operator=(const ComputeContext &other) = delete;

            VkInstance Instance() const;
            VkPhysicalDevice PhysicalDevice() const;
            VkDevice Device() const;
            VkQueue ComputeQueue() const;
            uint32_t ComputeQueueFamilyIndex() const;

            VmaAllocator Allocator() const;

        private:
            void Destroy();

            void CreateInstance();
            void SelectPhysicalDevice();
            void CreateLogicalDevice();
            void CreateAllocator();

            VkInstance m_instance;
            VkPhysicalDevice m_physicalDevice;
            VkDevice m_device;
            VkQueue m_computeQueue;
            uint32_t m_computeQueueFamilyIndex;
            VmaAllocator m_allocator;
        };
    }
}
