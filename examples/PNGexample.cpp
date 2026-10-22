// Headless: fills a small grid with a greyscale ramp and writes it to
// pngexample.png.

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <exception>
#include <string>

int main()
{
    try
    {
        const int32_t dim = 10;
        Hal::Examples::GridWindow win("PNG Example", dim, dim, 1, false);
        int32_t length = dim * dim;
        for (int32_t i = 0; i < length; i++)
        {
            win.SetPix(i, Hal::GreyScale(i * 1.0 / length));
        }
        win.ToPng("pngexample.png");
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("PNGexample failed: ") + e.what());
        return 1;
    }
}
