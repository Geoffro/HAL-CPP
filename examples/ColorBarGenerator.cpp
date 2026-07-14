// Ported from Examples/ColorBarGenerator: renders a labelled color bar for a
// heat map. Java drives it from an interactive menu (width/height/min/max/
// palette) and Util.DrawColorBar adds numeric tick labels; this port draws
// one vertical bar per palette side by side over a fixed value range and
// exports a PNG. Tick-label text is omitted (GridWindow has no text
// rasterizer yet).
//
// Live window by default; pass "headless" to only write colorbar.png.

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t barWidth = 20;
        const int32_t gap = 4;
        const int32_t height = 200;
        const double  vMin = 0.0;
        const double  vMax = 1.0;

        // One column of bars per palette.
        std::vector<int32_t (*)(double, double, double)> palettes = {
            Hal::HeatMapRgb, Hal::HeatMapRbg, Hal::HeatMapGrb, Hal::HeatMapGbr,
            Hal::HeatMapBrg, Hal::HeatMapBgr, Hal::HeatMapJet};

        int32_t nBars = static_cast<int32_t>(palettes.size());
        int32_t width = nBars * (barWidth + gap) + gap;

        Hal::Examples::GridWindow win("Color Bars", width, height, headless ? 2 : 3, !headless);
        win.Clear(Hal::BLACK);

        for (int32_t b = 0; b < nBars; b++)
        {
            int32_t x0 = gap + b * (barWidth + gap);
            for (int32_t y = 0; y < height; y++)
            {
                double  val = vMin + (vMax - vMin) * (y / static_cast<double>(height - 1));
                int32_t color = palettes[b](val, vMin, vMax);
                for (int32_t x = 0; x < barWidth; x++)
                {
                    win.SetPix(x0 + x, y, color);
                }
            }
        }
        win.Tick();
        win.ToPng("colorbar.png");

        if (!headless)
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
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("ColorBarGenerator failed: ") + e.what());
        return 1;
    }
}
