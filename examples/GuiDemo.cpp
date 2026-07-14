// Manual-verification demo for the windowed GUI stack: opens a window,
// animates a Grid2D as a texture filling it, and overlays a Dear ImGui
// control panel plus an ImPlot line plot. Not part of the automated test
// suite (rendering needs a live display) — build with HAL_BUILD_WINDOW and
// HAL_BUILD_GUI on and run it to check the on-screen result.

#include "Hal/Color.h"
#include "Hal/Grids/Grid2D.h"
#include "Hal/Gpu/Window.h"
#include "Hal/Gpu/WindowRenderContext.h"
#include "Hal/Gui/GuiContext.h"
#include "Hal/Gui/WindowGuiRenderer.h"
#include "Hal/Log/Logger.h"
#include "Hal/Video/GridRender.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <vector>

int main()
{
    try
    {
        const int32_t gridDim = 128;

        Hal::Gpu::Window window(960, 720, "HAL GUI Demo");
        Hal::Gpu::WindowRenderContext renderContext(window, gridDim, gridDim);
        Hal::Gui::GuiContext gui;
        Hal::Gui::WindowGuiRenderer guiRenderer(window, renderContext, gui);

        Hal::Grid2D<double> grid(gridDim, gridDim);

        bool paused = false;
        double speed = 1.0;
        double contrast = 0.5;
        int32_t frame = 0;

        std::vector<double> plotT;
        std::vector<double> plotV;

        while (!window.ShouldClose())
        {
            window.PollEvents();

            if (!paused)
            {
                double phase = static_cast<double>(frame) * 0.05 * speed;
                for (int32_t x = 0; x < gridDim; x++)
                {
                    for (int32_t y = 0; y < gridDim; y++)
                    {
                        double radial = std::sqrt(static_cast<double>((x - gridDim / 2) * (x - gridDim / 2) + (y - gridDim / 2) * (y - gridDim / 2)));
                        double value = 0.5 + contrast * std::sin(phase - radial * 0.2);
                        grid.Set(x, y, value);
                    }
                }

                plotT.push_back(static_cast<double>(frame));
                plotV.push_back(grid.Get(gridDim / 2, gridDim / 2));
                if (plotT.size() > 200)
                {
                    plotT.erase(plotT.begin());
                    plotV.erase(plotV.begin());
                }
                frame++;
            }

            guiRenderer.NewFrame();
            if (gui.BeginPanel("HAL Controls"))
            {
                gui.Text("Animated grid + ImGui + ImPlot");
                gui.Checkbox("paused", paused);
                gui.DoubleSlider("speed", speed, 0.0, 4.0);
                gui.DoubleSlider("contrast", contrast, 0.0, 0.5);
                if ((plotT.size() >= 2) && gui.BeginPlot("center cell", "frame", "value"))
                {
                    gui.PlotLine("value", plotT.data(), plotV.data(), static_cast<int32_t>(plotT.size()));
                    gui.EndPlot();
                }
            }
            gui.EndPanel();

            auto colorFn = [](double value)
            {
                return Hal::HeatMapJet(value, 0.0, 1.0);
            };
            renderContext.UpdateTexture(Hal::RenderGrid2DToRgba(grid, colorFn));

            bool presented = renderContext.DrawFrame([&](VkCommandBuffer commandBuffer)
            {
                guiRenderer.RenderOverlay(commandBuffer);
            });
            if (!presented)
            {
                // Frame skipped (resize recreated the swapchain, or the
                // window is minimized) — keep going, the next frame
                // draws normally.
            }
        }
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GuiDemo failed: ") + e.what());
        return 1;
    }
    return 0;
}
