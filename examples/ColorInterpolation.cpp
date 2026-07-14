// Ported from LEARN_HERE/Other/ColorInterpolation: bilinearly interpolates
// four corner colors across the grid (Interpolate2D per RGB component),
// producing a smooth 2D color field. Always writes colorinterpolation.png;
// also shows a window unless run with "headless".

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"
#include "Hal/MathUtil.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

static const double c1[3] = {0.0, 113.9850, 188.9550};
static const double c2[3] = {216.75, 82.875, 24.99};
static const double c3[3] = {236.895, 176.97, 31.875};
static const double c4[3] = {125.97, 46.92, 141.78};

static double InterpComponent(int32_t x,
                              int32_t y,
                              int32_t xDim,
                              int32_t yDim,
                              int32_t rgbIndex)
{
    return Hal::Interpolate2D(x * 1.0 / xDim, y * 1.0 / yDim,
                              c2[rgbIndex], c4[rgbIndex], c1[rgbIndex], c3[rgbIndex]);
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t dim = 100;
        Hal::Examples::GridWindow win("Color Interpolation", dim, dim, 5, !headless);
        for (int32_t x = 0; x < dim; x++)
        {
            for (int32_t y = 0; y < dim; y++)
            {
                int32_t r = static_cast<int32_t>(InterpComponent(x, y, dim, dim, 0));
                int32_t g = static_cast<int32_t>(InterpComponent(x, y, dim, dim, 1));
                int32_t b = static_cast<int32_t>(InterpComponent(x, y, dim, dim, 2));
                win.SetPix(x, y, Hal::Rgb256(r, g, b));
            }
        }
        win.Tick();
        win.ToPng("colorinterpolation.png");

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
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("ColorInterpolation failed: ") + e.what());
        return 1;
    }
}
