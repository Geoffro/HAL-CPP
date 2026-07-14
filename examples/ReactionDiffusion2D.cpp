// Ported from LEARN_HERE/Diffusibles/ReactionDiffusion2D.java: a source
// column feeds two PDE fields each tick; one is transported by advection,
// the other spread by (implicit ADI) diffusion, shown side by side. The
// Java version uses two UIGrids in one window; here they share a single
// combined window (advection on the left, diffusion on the right, one blank
// separator column between).
//
// Live window by default; pass "headless" to run without a window and dump
// the final frame to reactiondiffusion2d.png.

#include "Hal/Color.h"
#include "Hal/Grids/PDEGrid2D.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t x = 20;
        const int32_t y = 10;
        const int32_t time = headless ? 400 : 10000;
        const int32_t visScale = 10;

        // Combined window: [0,x) advection, x separator, [x+1, 2x+1) diffusion.
        const int32_t combinedW = x * 2 + 1;
        Hal::Examples::GridWindow win("2D Reaction-Diffusion", combinedW, y, visScale, !headless);

        Hal::PDEGrid2D g1(x, y); // advection
        Hal::PDEGrid2D g2(x, y); // diffusion

        for (int32_t i = 0; (i < time) && !win.IsClosed(); i++)
        {
            // reaction: hold a source column at 1.
            for (int32_t j = 3; j < (y - 3); j++)
            {
                g1.Set(10, j, 1.0);
                g2.Set(10, j, 1.0);
            }
            // advection
            g1.Advection(0.01, 0.01, 0.0);
            g1.Update();
            // diffusion
            g2.DiffusionADI(0.01);
            g2.Update();

            // draw both fields.
            for (int32_t px = 0; px < x; px++)
            {
                for (int32_t py = 0; py < y; py++)
                {
                    win.SetPix(px, py, Hal::HeatMapRgb(g1.Get(px, py)));
                    win.SetPix(x + 1 + px, py, Hal::HeatMapRgb(g2.Get(px, py)));
                }
            }
            win.TickPause(headless ? 0 : 5);
        }

        if (headless)
        {
            win.ToPng("reactiondiffusion2d.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("ReactionDiffusion2D failed: ") + e.what());
        return 1;
    }
}
