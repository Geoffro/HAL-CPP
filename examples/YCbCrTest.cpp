// Renders the Cb/Cr chroma plane (at fixed luma) across the grid.
//
// Live window by default; pass "headless" to dump the plane to ycbcrtest.png.

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

        const int32_t dim = 100;
        Hal::Examples::GridWindow win("CbCr Plane", dim, dim, 10, !headless);
        for (int32_t x = 0; x < dim; x++)
        {
            for (int32_t y = 0; y < dim; y++)
            {
                win.SetPix(x, y, Hal::CbCrPlaneColor(x * 1.0 / dim, y * 1.0 / dim));
            }
        }
        win.Tick();

        if (headless)
        {
            win.ToPng("ycbcrtest.png");
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
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("YCbCrTest failed: ") + e.what());
        return 1;
    }
}
