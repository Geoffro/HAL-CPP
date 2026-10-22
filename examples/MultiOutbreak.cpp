// A 7x3 grid of independent outbreak simulations, each with a different
// quarantine rate (increasing across rows), composited via
// MultiWellExperiment so the effect of quarantine can be compared at a
// glance.
//
// Live window by default; pass "headless" to run the fixed number of steps
// and dump the final composite to multioutbreak.png.

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"
#include "Hal/Tools/MultiWellExperiment.h"

#include "common/GridWindow.h"
#include "models/OutbreakWorld.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using Hal::Examples::OutbreakWorld;
using Hal::Examples::Person;

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t xDim = 7;
        const int32_t yDim = 3;

        std::vector<std::unique_ptr<OutbreakWorld>> owned;
        std::vector<OutbreakWorld *>                wells;
        for (int32_t row = 0; row < xDim; row++)
        {
            for (int32_t col = 0; col < yDim; col++)
            {
                auto model = std::make_unique<OutbreakWorld>();
                model->QUARANTINE_RATE_SYMPTOMATIC = row / 20.0;
                model->QUARANTINE_RATE_ASYMPTOMATIC = row / 20.0;
                wells.push_back(model.get());
                owned.push_back(std::move(model));
            }
        }

        Hal::MultiWellExperiment<OutbreakWorld> expt(xDim, yDim, OutbreakWorld::DIMENSION, OutbreakWorld::DIMENSION,
                                                     Hal::WHITE, wells);

        Hal::Examples::GridWindow win("Multi-Well Outbreak",
                                      expt.CompositeWidth(), expt.CompositeHeight(),
                                      headless ? 1 : 2, !headless);

        auto stepper = [](OutbreakWorld &model, int32_t) { model.StepCells(); };
        auto drawer = [](const OutbreakWorld &model, int32_t x, int32_t y) {
            const Person *pPerson = model.GetAgent(x, y);
            return OutbreakWorld::ReturnColor(pPerson->Type());
        };

        for (int32_t i = 0; (i < 80) && !win.IsClosed(); i++)
        {
            expt.Step(stepper);
            expt.Render(drawer);
            win.SetPixels(expt.Pixels());
            win.TickPause(headless ? 0 : 40);
        }

        if (headless)
        {
            win.ToPng("multioutbreak.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("MultiOutbreak failed: ") + e.what());
        return 1;
    }
}
