// Ported from Examples/GameOfLife/GOLSweepEx: a parameter sweep that runs
// many Game-of-Life simulations in parallel (one per random starting
// density) and writes each run's live-cell count over time to a CSV. The
// point of the example is the parallel sweep (Hal::MultiThread) plus
// per-run output collection, so the Life model here is a compact array
// version rather than the agent-based GOLGrid — it computes the identical
// Conway rule and per-tick live count.
//
// Headless: writes SweepResults.csv and prints progress.

#include "Hal/Concurrency/MultiThread.h"
#include "Hal/Rand.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

// One Conway step on a toroidal grid; returns the number of live cells.
static int32_t StepLife(std::vector<uint8_t> &cells,
                        int32_t               xDim,
                        int32_t               yDim)
{
    std::vector<uint8_t> next(cells.size(), 0);
    int32_t              liveCt = 0;
    for (int32_t x = 0; x < xDim; x++)
    {
        for (int32_t y = 0; y < yDim; y++)
        {
            int32_t n = 0;
            for (int32_t dx = -1; dx <= 1; dx++)
            {
                for (int32_t dy = -1; dy <= 1; dy++)
                {
                    if ((dx == 0) && (dy == 0))
                    {
                        continue;
                    }
                    int32_t nx = (x + dx + xDim) % xDim;
                    int32_t ny = (y + dy + yDim) % yDim;
                    n += cells[nx * yDim + ny];
                }
            }
            bool alive = cells[x * yDim + y] != 0;
            bool live = (alive && ((n == 2) || (n == 3))) || (!alive && (n == 3));
            next[x * yDim + y] = live ? 1 : 0;
            if (live)
            {
                liveCt++;
            }
        }
    }
    cells.swap(next);
    return liveCt;
}

int main()
{
    const int32_t nRuns = 100;
    const int32_t xDim = 100;
    const int32_t yDim = 100;
    const int32_t runTicks = 1000;

    std::vector<std::vector<double>> runOutputs(nRuns);

    Hal::MultiThread(nRuns, 8, [&](int32_t iRun) {
        Hal::Rand rng;
        double    startLivingProb = rng.Double();

        std::vector<uint8_t> cells(static_cast<size_t>(xDim) * yDim, 0);
        for (uint8_t &c : cells)
        {
            c = (rng.Double() < startLivingProb) ? 1 : 0;
        }

        std::vector<double> ret(runTicks + 1);
        ret[0] = startLivingProb;
        for (int32_t i = 0; i < runTicks; i++)
        {
            ret[i + 1] = StepLife(cells, xDim, yDim);
        }
        runOutputs[iRun] = std::move(ret);
    });

    std::ofstream out("SweepResults.csv");
    out << "startProp,popAtEachStep\n";
    for (const std::vector<double> &ret : runOutputs)
    {
        for (size_t i = 0; i < ret.size(); i++)
        {
            out << ret[i];
            if (i + 1 < ret.size())
            {
                out << ",";
            }
        }
        out << "\n";
    }
    out.close();

    std::printf("ran %d GoL sweeps -> SweepResults.csv\n", nRuns);
    return 0;
}
