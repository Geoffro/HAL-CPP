#include <doctest/doctest.h>

#include "GpuTestSupport.h"

#include "Hal/Color.h"
#include "Hal/Gpu/Scene3DRenderContext.h"
#include "Hal/Gpu/Window.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdint>
#include <memory>

TEST_CASE("Hal::Gpu::Scene3DRenderContext depth-tests and presents queued spheres and lines for several frames")
{
    std::unique_ptr<Hal::Gpu::Window> pWindow = TestHelpers::TryCreateWindow(320, 240, "HalCpp Scene3DRenderContext test");
    if (pWindow == nullptr)
    {
        return;
    }
    Hal::Gpu::Window              &window = *pWindow;
    Hal::Gpu::Scene3DRenderContext scene(window, 20.0, 10.0, 10.0);

    for (int32_t frame = 0; frame < 5; frame++)
    {
        window.PollEvents();
        scene.HandleInput();

        scene.ClearBox(Hal::Rgb(0.2, 0.2, 0.2), Hal::WHITE);
        double t = static_cast<double>(frame) * 0.3;
        scene.Sphere(10.0 + 3.0 * std::cos(t), 5.0, 5.0 + 3.0 * std::sin(t), 2.0, Hal::Rgb(1.0, 0.4, 0.1));
        scene.Sphere(10.0, 5.0, 5.0, 1.5, Hal::Rgb(0.1, 0.5, 1.0));
        scene.Line(0.0, 0.0, 0.0, 20.0, 10.0, 10.0, Hal::YELLOW);

        bool presented = scene.Render();
        CHECK(presented);
        if (window.ShouldClose())
        {
            break;
        }
    }

    // Exceed the initial instance capacity in one frame to exercise
    // the buffer-growth path (device idles, buffer is recreated
    // doubled, and the frame still presents).
    window.PollEvents();
    scene.Clear(Hal::BLACK);
    for (int32_t i = 0; i < 1500; i++)
    {
        double angle = static_cast<double>(i) * 0.02;
        scene.Sphere(10.0 + 5.0 * std::cos(angle),
                     5.0 + 3.0 * std::sin(angle * 1.7),
                     5.0 + 5.0 * std::sin(angle),
                     0.2,
                     Hal::HeatMapJet(static_cast<double>(i) / 1500.0, 0.0, 1.0));
    }
    bool presentedGrown = scene.Render();
    CHECK(presentedGrown);

    // Resize the window mid-run: the swapchain must be recreated
    // (via the resize flag or an OUT_OF_DATE result) and frames must
    // present again at the new size within a few attempts.
    glfwSetWindowSize(window.Handle(), 400, 300);
    bool presentedAfterResize = false;
    for (int32_t frame = 0; (frame < 5) && !presentedAfterResize; frame++)
    {
        window.PollEvents();
        scene.ClearBox(Hal::Rgb(0.2, 0.2, 0.2), Hal::WHITE);
        scene.Sphere(10.0, 5.0, 5.0, 2.0, Hal::Rgb(1.0, 0.4, 0.1));
        presentedAfterResize = scene.Render();
    }
    CHECK(presentedAfterResize);
}
