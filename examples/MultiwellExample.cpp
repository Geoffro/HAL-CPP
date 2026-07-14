// Ported from LEARN_HERE/Agents/MultiwellExample: six independent BirthDeath
// wells (one per named color) run side by side in a 3x2 grid via
// MultiWellExperiment. The C++ MultiWellExperiment composites the wells into
// one pixel buffer each frame (rather than driving a live Swing grid), which
// we blit into a GridWindow.
//
// Live window by default; pass "headless" to run fewer steps and dump the
// final composite to multiwellexample.png.

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"
#include "Hal/Tools/MultiWellExperiment.h"

#include "common/GridWindow.h"
#include "models/BirthDeathModel.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using Hal::Examples::BirthDeath;
using Hal::Examples::Cell;

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t x = 100;
        const int32_t y = 100;
        const int32_t colors[6] = {Hal::RED, Hal::GREEN, Hal::BLUE, Hal::YELLOW, Hal::CYAN, Hal::MAGENTA};

        std::vector<std::unique_ptr<BirthDeath>> owned;
        std::vector<BirthDeath *>                wells;
        for (int32_t i = 0; i < 6; i++)
        {
            owned.push_back(std::make_unique<BirthDeath>(x, y, colors[i]));
            owned.back()->Setup(2);
            wells.push_back(owned.back().get());
        }

        Hal::MultiWellExperiment<BirthDeath> expt(3, 2, x, y, Hal::WHITE, wells);

        Hal::Examples::GridWindow win("Multi-Well BirthDeath",
                                      expt.CompositeWidth(), expt.CompositeHeight(),
                                      headless ? 2 : 3, !headless);

        auto stepper = [](BirthDeath &model, int32_t) { model.Step(); };
        auto drawer = [](const BirthDeath &model, int32_t px, int32_t py) {
            const Cell *pCell = model.GetAgent(px, py);
            return (pCell == nullptr) ? Hal::BLACK : pCell->Color();
        };

        int32_t steps = headless ? 200 : 100000;
        for (int32_t i = 0; (i < steps) && !win.IsClosed(); i++)
        {
            expt.Step(stepper);
            expt.Render(drawer);
            win.SetPixels(expt.Pixels());
            win.TickPause(headless ? 0 : 30);
        }

        if (headless)
        {
            win.ToPng("multiwellexample.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("MultiwellExample failed: ") + e.what());
        return 1;
    }
}
