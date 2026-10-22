#pragma once

#include "Hal/Gpu/Window.h"

#include <volk.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace Hal
{
    namespace Gpu
    {
        // Presents a fixed-size RGBA8 texture, updated per frame, across the
        // whole window via a full-screen triangle (FullscreenQuad.vert/.frag).
        // Owns its own Vulkan instance and device, separate from ComputeContext.
        // One frame in flight. The swapchain is rebuilt on resize or OUT_OF_DATE,
        // and frames are skipped while the window is minimized.
        class WindowRenderContext
        {
        public:
            WindowRenderContext(Window  &window,
                                int32_t  textureWidth,
                                int32_t  textureHeight);
            ~WindowRenderContext();

            WindowRenderContext(const WindowRenderContext &other) = delete;
            WindowRenderContext &operator=(const WindowRenderContext &other) = delete;

            // pixelsRgba must be textureWidth*textureHeight RGBA8 (row 0 at the top,
            // as RenderGrid2DToRgba produces); throws std::invalid_argument
            // otherwise. Blocks until the upload completes.
            void UpdateTexture(const std::vector<uint8_t> &pixelsRgba);

            // Draws the texture, then overlay(commandBuffer) in the same render
            // pass (e.g. for an ImGui layer), and presents. Returns false if the
            // frame was skipped (swapchain recreated or window minimized); keep
            // calling.
            bool DrawFrame(const std::function<void(VkCommandBuffer)> &overlay);
            bool DrawFrame();

            // Handles an overlay renderer (e.g. imgui_impl_vulkan) initializes
            // against; valid for this object's lifetime.
            VkInstance Instance() const;
            VkPhysicalDevice PhysicalDevice() const;
            VkDevice Device() const;
            uint32_t GraphicsQueueFamilyIndex() const;
            VkQueue GraphicsQueue() const;
            VkRenderPass RenderPass() const;
            uint32_t SwapchainImageCount() const;

        private:
            void Destroy();

            void CreateInstance();
            void CreateSurface();
            void SelectPhysicalDevice();
            void CreateLogicalDevice();
            void CreateSwapchain();
            void CreateRenderPass();
            void CreateFramebuffers();
            void CreateTextureResources();
            void CreateGraphicsPipeline();
            void CreateCommandObjects();
            void CreateSyncObjects();

            VkCommandBuffer BeginSingleTimeCommands();
            void EndSingleTimeCommands(VkCommandBuffer commandBuffer);

            // Rebuilds the swapchain, image views, and framebuffers. Returns false
            // without rebuilding while the framebuffer is zero-sized (minimized).
            bool RecreateSwapchain();

            Window &m_window;
            int32_t m_textureWidth;
            int32_t m_textureHeight;

            VkInstance m_instance;
            VkSurfaceKHR m_surface;
            VkPhysicalDevice m_physicalDevice;
            VkDevice m_device;
            uint32_t m_graphicsQueueFamilyIndex;
            uint32_t m_presentQueueFamilyIndex;
            VkQueue m_graphicsQueue;
            VkQueue m_presentQueue;

            VkSwapchainKHR m_swapchain;
            VkFormat m_swapchainFormat;
            VkExtent2D m_swapchainExtent;
            std::vector<VkImage> m_swapchainImages;
            std::vector<VkImageView> m_swapchainImageViews;

            VkRenderPass m_renderPass;
            std::vector<VkFramebuffer> m_framebuffers;

            VkImage m_textureImage;
            VkDeviceMemory m_textureMemory;
            VkImageView m_textureImageView;
            VkSampler m_textureSampler;
            VkBuffer m_stagingBuffer;
            VkDeviceMemory m_stagingMemory;

            VkDescriptorSetLayout m_descriptorSetLayout;
            VkDescriptorPool m_descriptorPool;
            VkDescriptorSet m_descriptorSet;
            VkShaderModule m_vertShaderModule;
            VkShaderModule m_fragShaderModule;
            VkPipelineLayout m_pipelineLayout;
            VkPipeline m_pipeline;

            VkCommandPool m_commandPool;
            std::vector<VkCommandBuffer> m_commandBuffers;

            VkSemaphore m_imageAvailableSemaphore;
            VkSemaphore m_renderFinishedSemaphore;
            VkFence m_inFlightFence;
        };
    }
}
