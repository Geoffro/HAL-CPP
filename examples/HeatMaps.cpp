// Paints the six channel-permutation heat maps (RGB, RBG, GRB, GBR, BRG,
// BGR) as horizontal ramps, one row each, to compare the palettes.
//
// Live window by default; pass "headless" to dump the swatches to
// heatmaps.png.

#include "Hal/Color.h"
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

        Hal::Examples::GridWindow win("Heat Maps", 100, 6, 10, !headless);
        for (int32_t j = 0; j < 100; j++)
        {
            double v = j * 1.0 / 100;
            win.SetPix(j, 5, Hal::HeatMapRgb(v));
            win.SetPix(j, 4, Hal::HeatMapRbg(v));
            win.SetPix(j, 3, Hal::HeatMapGrb(v));
            win.SetPix(j, 2, Hal::HeatMapGbr(v));
            win.SetPix(j, 1, Hal::HeatMapBrg(v));
            win.SetPix(j, 0, Hal::HeatMapBgr(v));
        }
        win.Tick();

        if (headless)
        {
            win.ToPng("heatmaps.png");
        }
        else
        {
            while (!win.IsClosed())
            {
                win.TickPause(50);
            }
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("HeatMaps failed: ") + e.what());
        return 1;
    }
}
