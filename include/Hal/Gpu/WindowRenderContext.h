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
        // Vulkan graphics pipeline that presents a live, per-frame-
        // updatable pixel-grid texture to a Window — the
        // `UIGrid`-equivalent live render surface, built on top of the
        // plain clear-and-present milestone that came before it.
        // Replaces Java's
        // `UIWindow`/`OpenGL2DWindow`/`OpenGL3DWindow`.
        //
        // Owns its own instance/device (separate from
        // Hal::Gpu::ComputeContext, which is deliberately surface-less) —
        // an instance/device unification between headless compute and
        // windowed rendering is a legitimate follow-up, not required for
        // this milestone.
        //
        // The texture is a fixed textureWidth x textureHeight RGBA8
        // image, sized once at construction (same "fixed-size resources,
        // correctness first" stance as DiffusionKernel2D/
        // AgentForceKernel2D). UpdateTexture() re-uploads its full
        // contents via a staging buffer each time it's called — no
        // partial/dirty-rect updates yet, matching HAL's own model loop
        // shape (recompute the whole visible grid each tick, then render
        // it). A full-screen triangle (FullscreenQuad.vert/.frag, no
        // vertex/index buffer needed) samples it, so the texture always
        // covers the entire window regardless of its own aspect ratio;
        // fitting/letterboxing is a legitimate follow-up.
        //
        // Single frame in flight (one fence, synchronous acquire →
        // record → submit → present) — the same "correctness first"
        // stance as every compute kernel in Phase 4. Window resizes are
        // handled: the swapchain (plus image views and framebuffers) is
        // rebuilt when the window reports a resize or an acquire/present
        // returns OUT_OF_DATE, and while the window is minimized (a
        // zero-size framebuffer) frames are skipped until it is restored.
        // The pipelines use dynamic viewport/scissor state so only the
        // swapchain-sized resources are recreated, never the pipeline.
        class WindowRenderContext
        {
        public:
            WindowRenderContext(Window  &window,
                                int32_t  textureWidth,
                                int32_t  textureHeight);
            ~WindowRenderContext();

            WindowRenderContext(const WindowRenderContext &other) = delete;
            WindowRenderContext &operator=(const WindowRenderContext &other) = delete;

            // pixelsRgba must be textureWidth*textureHeight RGBA8 (4
            // bytes/pixel, row-major, top-to-bottom — matching
            // Hal::RenderGrid2DToRgba's output). Blocks until the upload
            // completes.
            void UpdateTexture(const std::vector<uint8_t> &pixelsRgba);

            // Acquires the next swapchain image, draws the current texture
            // contents over the full window, then (if provided) invokes
            // overlay(commandBuffer) inside the same render pass so a UI
            // layer (e.g. Dear ImGui) can draw on top, and presents.
            // Returns true if a frame was presented; false if this frame
            // was skipped because the swapchain had to be recreated
            // (window resize) or the window is minimized — keep calling,
            // the next frame draws normally.
            bool DrawFrame(const std::function<void(VkCommandBuffer)> &overlay);
            bool DrawFrame();

            // Vulkan handles a UI/overlay renderer (e.g. imgui_impl_vulkan)
            // needs to initialize against this context's device and render
            // pass. All owned by this object and valid for its lifetime.
            VkInstance Instance() const;
            VkPhysicalDevice PhysicalDevice() const;
            VkDevice Device() const;
            uint32_t GraphicsQueueFamilyIndex() const;
            VkQueue GraphicsQueue() const;
            VkRenderPass RenderPass() const;
            uint32_t SwapchainImageCount() const;

        private:
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

            // Rebuilds the swapchain-sized resources (swapchain, image
            // views, framebuffers) after a resize. Returns false without
            // recreating anything while the framebuffer is zero-sized
            // (minimized) — callers skip the frame and retry later.
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
