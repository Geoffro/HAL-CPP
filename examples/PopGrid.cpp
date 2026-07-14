// Ported from LEARN_HERE/Agents/PopGrid: a count-based (mean-field)
// diffusion of a large population across a PopulationGrid2D. Each occupied
// cell hands a multinomial share of its count to each von-Neumann neighbor,
// so an initial point mass spreads out like a random walk. Counts move
// through the delta buffer and are committed by Update() each tick.
//
// Live window by default; pass "headless" to run a bounded number of steps
// and dump the final frame to popgrid.png.

#include "Hal/Color.h"
#include "Hal/Grids/PopulationGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"
#include "Hal/Tools/MultinomialCalc.h"

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

        Hal::PopulationGrid2D<int32_t> cells(50, 50);
        Hal::Rand                      rng;
        Hal::MultinomialCalc           mn(rng);
        Hal::Examples::GridWindow      win("Population Grid", cells.Xdim(), cells.Ydim(), 10, !headless);

        cells.Set(0, 0, 100000000);
        std::vector<int32_t> hood = Hal::VonNeumannHood(false);

        int32_t maxSteps = headless ? 300 : 1000000;
        for (int32_t step = 0; (step < maxSteps) && !win.IsClosed(); step++)
        {
            cells.Update();

            for (int32_t i : cells)
            {
                int32_t numMoves = cells.MapHood(hood.data(), static_cast<int32_t>(hood.size()), i);
                mn.Setup(cells.Get(i));
                for (int32_t j = 0; j < numMoves; j++)
                {
                    int32_t movePop = mn.Sample(0.1);
                    cells.Add(i, -movePop);
                    cells.Add(hood[j], movePop);
                }
            }

            win.TickPause(0);
            int32_t length = cells.Xdim() * cells.Ydim();
            for (int32_t i = 0; i < length; i++)
            {
                win.SetPix(i, Hal::HeatMapRgb(cells.Get(i) / 100000.0));
            }
        }

        if (headless)
        {
            win.ToPng("popgrid.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("PopGrid failed: ") + e.what());
        return 1;
    }
}
