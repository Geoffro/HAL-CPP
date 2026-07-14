#pragma once

#include <volk.h>

#include <cstdint>

// Forward declaration of VMA's allocator handle (declared the same way in
// vk_mem_alloc.h — an identical typedef is legal), so this public header
// doesn't pull the whole VMA header in; only Hal::Gpu .cpp files include
// vk_mem_alloc.h.
VK_DEFINE_HANDLE(VmaAllocator)

namespace Hal
{
    namespace Gpu
    {
        // Minimal headless Vulkan compute context: instance + physical
        // device + logical device + compute queue, no swapchain/surface.
        // This is the foundation GPU compute kernels (diffusion, agent
        // neighbor search) get built on top of.
        //
        // Non-copyable: owns Vulkan handles with manual RAII cleanup in
        // the destructor (reverse creation order). This is the case
        // coding_standards.md's "avoid smart-pointer churn" carves room
        // for — a single owner with a strict creation/teardown order is
        // simplest expressed as plain handles plus an explicit
        // destructor, not a smart pointer.
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

            // The VMA allocator every compute kernel allocates its
            // buffers through (sub-allocation and mapping management
            // replace the earlier manual vkAllocateMemory-per-buffer
            // path). Owned by this context; kernels must be destroyed
            // before it.
            VmaAllocator Allocator() const;

        private:
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
