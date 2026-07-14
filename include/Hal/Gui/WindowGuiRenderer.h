#pragma once

#include "Hal/Gpu/Window.h"
#include "Hal/Gpu/WindowRenderContext.h"
#include "Hal/Gui/GuiContext.h"

#include <volk.h>

namespace Hal
{
    namespace Gui
    {
        // Renders a GuiContext's UI onto a Hal::Gpu::Window, wiring Dear
        // ImGui's GLFW input backend (imgui_impl_glfw) and Vulkan rendering
        // backend (imgui_impl_vulkan) to the window and its render context.
        // This is the on-screen counterpart to the headless GuiContext: it
        // turns built UI into actual pixels over the window's swapchain.
        //
        // Lifetime: construct after the Window and WindowRenderContext, and
        // destroy before them — it holds references and its teardown uses
        // their Vulkan device. Not covered by the automated test suite
        // (rendering needs a live display); verify on a real machine.
        //
        // Per-frame usage:
        //   guiRenderer.NewFrame();
        //   gui.BeginPanel(...); /* build UI */ gui.EndPanel();
        //   renderContext.UpdateTexture(pixels);
        //   renderContext.DrawFrame([&](VkCommandBuffer cmd)
        //   {
        //       guiRenderer.RenderOverlay(cmd);
        //   });
        class WindowGuiRenderer
        {
        public:
            WindowGuiRenderer(Gpu::Window              &window,
                              Gpu::WindowRenderContext &renderContext,
                              GuiContext               &gui);
            ~WindowGuiRenderer();

            WindowGuiRenderer(const WindowGuiRenderer &other) = delete;
            WindowGuiRenderer &operator=(const WindowGuiRenderer &other) = delete;

            // Advances the ImGui GLFW + Vulkan backends and starts a new UI
            // frame (the backends set display size / delta time / input).
            // Build the UI via the GuiContext after this.
            void NewFrame();

            // Finalizes the UI frame and records its draw commands into
            // commandBuffer — pass this as the overlay to
            // WindowRenderContext::DrawFrame.
            void RenderOverlay(VkCommandBuffer commandBuffer);

        private:
            Gpu::Window &m_window;
            Gpu::WindowRenderContext &m_renderContext;
            GuiContext &m_gui;
            VkDescriptorPool m_descriptorPool;
        };
    }
}
