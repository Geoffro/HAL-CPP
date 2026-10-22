#pragma once

#include "Hal/Gpu/Window.h"
#include "Hal/Gpu/WindowRenderContext.h"
#include "Hal/Gui/GuiContext.h"

#include <volk.h>

namespace Hal
{
    namespace Gui
    {
        // Draws a GuiContext's UI over a Window through ImGui's GLFW and Vulkan
        // backends. Construct after the Window and WindowRenderContext and
        // destroy before them. Not covered by the tests (needs a display).
        //
        // Per frame:
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

            // Finalizes the UI frame and records its draw commands; pass as the
            // overlay to WindowRenderContext::DrawFrame.
            void RenderOverlay(VkCommandBuffer commandBuffer);

        private:
            Gpu::Window &m_window;
            Gpu::WindowRenderContext &m_renderContext;
            GuiContext &m_gui;
            VkDescriptorPool m_descriptorPool;
        };
    }
}
