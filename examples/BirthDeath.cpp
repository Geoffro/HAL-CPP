// A stochastic birth/death model on an AgentGrid2D seeded with a disc of
// cells. Each tick a cell may die (freeing its square) or divide into a
// random empty Moore neighbor. The model lives in models/BirthDeathModel.h
// (shared with MultiwellExample); this file adds the visualization and run
// loop.
//
// Live window by default; pass "headless" to run without a window and dump
// the final state to birthdeath.png.

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"
#include "models/BirthDeathModel.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

namespace BirthDeathExample
{
    void DrawModel(Hal::Examples::BirthDeath &model,
                   Hal::Examples::GridWindow &vis)
    {
        for (int32_t i = 0; i < model.Length(); i++)
        {
            Hal::Examples::Cell *pCell = model.GetAgent(i);
            vis.SetPix(i, pCell == nullptr ? Hal::BLACK : pCell->Color());
        }
    }
}
using namespace BirthDeathExample;

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        Hal::Examples::BirthDeath model(100, 100, Hal::RED);
        Hal::Examples::GridWindow win("Birth / Death", 100, 100, 8, !headless);
        model.Setup(10);

        int32_t steps = headless ? 500 : 100000;
        for (int32_t i = 0; (i < steps) && !win.IsClosed(); i++)
        {
            win.TickPause(headless ? 0 : 10);
            model.Step();
            DrawModel(model, win);
        }

        if (headless)
        {
            win.ToPng("birthdeath.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("BirthDeath failed: ") + e.what());
        return 1;
    }
}
