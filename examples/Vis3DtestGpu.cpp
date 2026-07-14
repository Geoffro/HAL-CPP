// Hardware-Vulkan counterpart of Vis3Dtest: the same wireframe box and
// axis spheres, rendered by Hal::Gpu::Scene3DRenderContext (real depth
// testing, instanced sphere meshes, Lambert lighting) instead of the
// software Scene3D rasterizer. A pair of orbiting spheres deliberately
// pass through each other — their intersection resolves per pixel via
// the depth buffer, which the software renderer's painter's algorithm
// cannot do.
//
// Interactive only (a live window is the Vulkan surface): drag to
// orbit, scroll to zoom, resize freely (the swapchain is recreated on
// the fly). Run it on a display to verify the renderer.

#include "Hal/Color.h"
#include "Hal/Gpu/Scene3DRenderContext.h"
#include "Hal/Gpu/Window.h"
#include "Hal/Log/Logger.h"

#include <chrono>
#include <cmath>
#include <exception>
#include <string>
#include <thread>

int main()
{
    try
    {
        const double xDim = 20.0;
        const double yDim = 10.0;
        const double zDim = 10.0;

        Hal::Gpu::Window               window(960, 720, "3D test (hardware Vulkan renderer)");
        Hal::Gpu::Scene3DRenderContext scene(window, xDim, yDim, zDim);

        double t = 0.0;
        while (!window.ShouldClose())
        {
            window.PollEvents();
            scene.HandleInput();

            scene.ClearBox(Hal::Rgb(0.5, 0.5, 0.5), Hal::WHITE);

            // The same three axis markers as Vis3Dtest.
            scene.Sphere(20.0, 0.0, 0.0, 1.0, Hal::Rgb(1, 0, 0));
            scene.Sphere(0.0, 30.0, 0.0, 1.0, Hal::Rgb(0, 1, 0));
            scene.Sphere(0.0, 0.0, 10.0, 1.0, Hal::Rgb(0, 0, 1));

            // Two orbiting spheres that periodically overlap: the depth
            // buffer draws their true intersection.
            double cx = xDim / 2.0;
            double cy = yDim / 2.0;
            double cz = zDim / 2.0;
            scene.Sphere(cx + 2.5 * std::cos(t), cy, cz + 2.5 * std::sin(t), 2.0, Hal::Rgb(1.0, 0.6, 0.1));
            scene.Sphere(cx + 2.5 * std::cos(t + 3.14159), cy + std::sin(t * 0.7), cz + 2.5 * std::sin(t + 3.14159), 2.0, Hal::Rgb(0.2, 0.5, 1.0));

            // A false return is a skipped frame (resize/minimize), not an
            // error — keep looping and the next frame draws normally.
            scene.Render();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            t += 0.02;
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("Vis3DtestGpu failed: ") + e.what());
        return 1;
    }
}
