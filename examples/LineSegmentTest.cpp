// Draws a single red 3D line segment across the diagonal of the domain
// with the software Scene3D renderer.
//
// Live window by default (drag to orbit); pass "headless" to render one
// frame to linesegmenttest.png.

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

        Hal::Examples::GridWindow win("segment", 500, 500, 1, !headless);
        Hal::Examples::Scene3D    scene(win, 10, 10, 10);

        auto drawFrame = [&]() {
            scene.HandleInput();
            scene.Clear(Hal::BLACK);
            scene.Line(0, 0, 0, 10, 10, 10, Hal::Rgb(1, 0, 0));
            scene.Render();
        };

        if (headless)
        {
            drawFrame();
            win.ToPng("linesegmenttest.png");
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
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("LineSegmentTest failed: ") + e.what());
        return 1;
    }
}
