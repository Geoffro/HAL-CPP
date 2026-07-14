#include "Hal/Gui/WindowGuiRenderer.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace Hal
{
    namespace Gui
    {
        WindowGuiRenderer::WindowGuiRenderer(Gpu::Window              &window,
                                             Gpu::WindowRenderContext &renderContext,
                                             GuiContext               &gui)
            : m_window(window),
              m_renderContext(renderContext),
              m_gui(gui),
              m_descriptorPool(VK_NULL_HANDLE)
        {
            m_gui.MakeCurrent();

            // ImGui's Vulkan backend needs a descriptor pool it can allocate
            // its font (and any user-added) textures from; it must allow
            // freeing individual sets.
            VkDescriptorPoolSize poolSize{};
            poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            poolSize.descriptorCount = 64;

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            poolInfo.maxSets = 64;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            Gpu::Internal::CheckResult(vkCreateDescriptorPool(m_renderContext.Device(), &poolInfo, nullptr, &m_descriptorPool), "vkCreateDescriptorPool (ImGui) failed");

            ImGui_ImplGlfw_InitForVulkan(m_window.Handle(), true);

            ImGui_ImplVulkan_InitInfo initInfo{};
            initInfo.Instance = m_renderContext.Instance();
            initInfo.PhysicalDevice = m_renderContext.PhysicalDevice();
            initInfo.Device = m_renderContext.Device();
            initInfo.QueueFamily = m_renderContext.GraphicsQueueFamilyIndex();
            initInfo.Queue = m_renderContext.GraphicsQueue();
            initInfo.DescriptorPool = m_descriptorPool;
            initInfo.RenderPass = m_renderContext.RenderPass();
            initInfo.MinImageCount = 2;
            initInfo.ImageCount = m_renderContext.SwapchainImageCount();
            initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
            ImGui_ImplVulkan_Init(&initInfo);

            // Upload the font atlas to a GPU texture (self-managed by the
            // backend in this ImGui version).
            ImGui_ImplVulkan_CreateFontsTexture();
        }

        WindowGuiRenderer::~WindowGuiRenderer()
        {
            vkDeviceWaitIdle(m_renderContext.Device());
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            if (m_descriptorPool != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(m_renderContext.Device(), m_descriptorPool, nullptr);
            }
        }

        void WindowGuiRenderer::NewFrame()
        {
            m_gui.MakeCurrent();
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            m_gui.NewFrameRaw();
        }

        void WindowGuiRenderer::RenderOverlay(VkCommandBuffer commandBuffer)
        {
            m_gui.EndFrame();
            ImGui_ImplVulkan_RenderDrawData(m_gui.DrawData(), commandBuffer);
        }
    }
}
