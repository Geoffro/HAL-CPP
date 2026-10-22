// One SIR-style outbreak spreading from a central cluster. Susceptible
// (grey) turns asymptomatic (pink), symptomatic (red), then recovered
// (green) or dead (black); quarantined individuals draw blue, and a green
// border flags when infections exceed hospital capacity.
//
// Live window by default; pass "headless" to run the fixed number of steps
// and dump the final frame to outbreakworld.png.

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"
#include "models/OutbreakWorld.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

using Hal::Examples::OutbreakWorld;
using Hal::Examples::Person;

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t dim = OutbreakWorld::DIMENSION;
        Hal::Examples::GridWindow win("Outbreak", dim, dim, headless ? 3 : 5, !headless);
        OutbreakWorld world;

        const int32_t totalTime = 50;
        for (int32_t t = 0; (t < totalTime) && !win.IsClosed(); t++)
        {
            for (int32_t i = 0; i < world.Length(); i++)
            {
                Person *pPerson = world.GetAgent(i);
                if (pPerson != nullptr)
                {
                    win.SetPix(i, pPerson->Quarantined() ? Hal::BLUE : OutbreakWorld::ReturnColor(pPerson->Type()));
                }
            }
            if (world.AboveCapacity())
            {
                for (int32_t i = 0; i < dim; i++)
                {
                    win.SetPix(0, i, Hal::GREEN);
                    win.SetPix(dim - 1, i, Hal::GREEN);
                    win.SetPix(i, 0, Hal::GREEN);
                    win.SetPix(i, dim - 1, Hal::GREEN);
                }
            }
            win.TickPause(headless ? 0 : 100);
            world.StepCells();
        }

        if (headless)
        {
            win.ToPng("outbreakworld.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("OutbreakWorld failed: ") + e.what());
        return 1;
    }
}
