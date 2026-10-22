#include <doctest/doctest.h>

#include "GpuTestSupport.h"

#include "Hal/Color.h"
#include "Hal/Grids/Grid2D.h"
#include "Hal/Gpu/Window.h"
#include "Hal/Gpu/WindowRenderContext.h"
#include "Hal/Video/GridRender.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

TEST_CASE("Hal::Gpu::WindowRenderContext opens a window and presents a live pixel-grid texture for several frames")
{
    const int32_t gridDim = 32;

    std::unique_ptr<Hal::Gpu::Window> pWindow = TestHelpers::TryCreateWindow(320, 240, "HalCpp WindowRenderContext test");
    if (pWindow == nullptr)
    {
        return;
    }
    Hal::Gpu::Window             &window = *pWindow;
    Hal::Gpu::WindowRenderContext renderContext(window, gridDim, gridDim);
    CHECK_THROWS_AS(renderContext.UpdateTexture(std::vector<uint8_t>(16, 0)), std::invalid_argument);

    Hal::Grid2D<double> grid(gridDim, gridDim);
    auto colorFn = [](double value)
    {
        return Hal::HeatMapJet(value, 0.0, 1.0);
    };

    for (int32_t frame = 0; frame < 5; frame++)
    {
        window.PollEvents();

        for (int32_t x = 0; x < gridDim; x++)
        {
            for (int32_t y = 0; y < gridDim; y++)
            {
                double phase = static_cast<double>(frame) * 0.2;
                grid.Set(x, y, 0.5 + 0.5 * std::sin(phase + static_cast<double>(x + y) / gridDim));
            }
        }

        renderContext.UpdateTexture(Hal::RenderGrid2DToRgba(grid, colorFn));
        bool presented = renderContext.DrawFrame();
        CHECK(presented);
        if (window.ShouldClose())
        {
            break;
        }
    }

    // Resize the window mid-run: the swapchain must be recreated
    // (via the resize flag or an OUT_OF_DATE result) and frames must
    // present again at the new size within a few attempts.
    glfwSetWindowSize(window.Handle(), 400, 300);
    bool presentedAfterResize = false;
    for (int32_t frame = 0; (frame < 5) && !presentedAfterResize; frame++)
    {
        window.PollEvents();
        presentedAfterResize = renderContext.DrawFrame();
    }
    CHECK(presentedAfterResize);
}
