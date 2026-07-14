// Ported from LEARN_HERE/Other/GIFtest: animate a single randomly-recolored
// pixel and record the frames to a movie. Java writes a GIF (GifMaker); this
// port uses the C++ video path (VideoWriter, MP4) that replaced GIF export,
// via GridWindow's recording API. Also dumps the final frame as a PNG.
//
// Headless by nature (its point is the output files).

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <exception>
#include <string>

int main()
{
    try
    {
        Hal::Rand                 rng;
        Hal::Examples::GridWindow win("gif test", 10, 10, 20, false);

        win.StartRecording("giftest.mp4", 10);
        for (int32_t i = 0; i < 10; i++)
        {
            win.SetPix(4, 4, Hal::Rgb(rng.Double(), rng.Double(), rng.Double()));
            win.RecordFrame();
        }
        win.StopRecording();
        win.ToPng("giftest.png");
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GIFtest failed: ") + e.what());
        return 1;
    }
}
