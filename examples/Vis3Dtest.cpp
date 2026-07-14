// Ported from LEARN_HERE/Other/vis3Dtest: a minimal 3D-window smoke test —
// three colored spheres placed along the axes inside a wireframe box. Uses
// the software Scene3D renderer.
//
// Live window by default (drag to orbit, up/down to zoom); pass "headless"
// to render one frame to vis3dtest.png.

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"
#include "common/Scene3D.h"

#include <cstring>
#include <exception>
#include <string>

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        Hal::Examples::GridWindow win("3D test", 640, 480, 1, !headless);
        Hal::Examples::Scene3D    scene(win, 20, 10, 10);

        auto drawFrame = [&]() {
            scene.HandleInput();
            scene.ClearBox(Hal::Rgb(0.5, 0.5, 0.5), Hal::WHITE);
            scene.Sphere(20, 0, 0, 1, Hal::Rgb(1, 0, 0));
            scene.Sphere(0, 30, 0, 1, Hal::Rgb(0, 1, 0));
            scene.Sphere(0, 0, 10, 1, Hal::Rgb(0, 0, 1));
            scene.Render();
        };

        if (headless)
        {
            drawFrame();
            win.ToPng("vis3dtest.png");
        }
        else
        {
            while (!win.IsClosed())
            {
                drawFrame();
                win.TickPause(16);
            }
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("Vis3Dtest failed: ") + e.what());
        return 1;
    }
}
