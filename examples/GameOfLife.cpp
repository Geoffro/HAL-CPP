// Conway's Game of Life on an AgentGrid2D. Each square holds one GOLAgent
// whose live/dead state is double-buffered by tick parity, so a whole
// generation is computed from the previous one before any cell flips.
//
// Live window by default; pass "headless" as the first argument to run
// without a window and dump the final generation to gameoflife.png.

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace GameOfLife
{
    const int32_t LIVE = Hal::Rgb(1.0, 0.0, 0.0);
    const int32_t DEAD = Hal::Rgb(0.0, 0.0, 0.0);
}
using namespace GameOfLife;

class GolGrid;

class GolAgent : public Hal::AgentSQ2DUnstackable<GolGrid>
{
public:
    explicit GolAgent(GolGrid *pGrid);

    int32_t GetCurrState() const;
    void SetNextState(int32_t state);
    void Step();

private:
    int32_t m_state[2];
};

class GolGrid : public Hal::AgentGrid2D<GolAgent, GolGrid>
{
public:
    GolGrid(int32_t xDim,
            int32_t yDim,
            double  livingProb,
            Hal::Rand &rng)
        : Hal::AgentGrid2D<GolAgent, GolGrid>(xDim, yDim, true, true),
          m_mooreHood(Hal::MooreHood(false)),
          m_liveCt(0)
    {
        for (int32_t i = 0; i < Length(); i++)
        {
            GolAgent &a = NewAgentSQ(i);
            a.SetNextState(rng.Double() < livingProb ? LIVE : DEAD);
        }
    }

    int32_t *MooreHoodData() { return m_mooreHood.data(); }
    int32_t  MooreHoodSize() const { return static_cast<int32_t>(m_mooreHood.size()); }
    int32_t  LiveCount() const { return m_liveCt; }

    void StepAgents()
    {
        IncTick();
        m_liveCt = 0;
        for (GolAgent &agent : *this)
        {
            agent.Step();
        }
    }

    void AddLive() { m_liveCt++; }

private:
    std::vector<int32_t> m_mooreHood;
    int32_t              m_liveCt;
};

GolAgent::GolAgent(GolGrid *pGrid)
    : Hal::AgentSQ2DUnstackable<GolGrid>(pGrid),
      m_state{DEAD, DEAD}
{
}

int32_t GolAgent::GetCurrState() const
{
    return m_state[this->Grid()->GetTick() % 2];
}

void GolAgent::SetNextState(int32_t state)
{
    m_state[(this->Grid()->GetTick() + 1) % 2] = state;
}

void GolAgent::Step()
{
    GolGrid *pGrid = this->Grid();
    int32_t  hoodLen = MapHood(pGrid->MooreHoodData(), pGrid->MooreHoodSize());
    int32_t  countNeighbors = 0;
    for (int32_t i = 0; i < hoodLen; i++)
    {
        if (pGrid->GetAgent(pGrid->MooreHoodData()[i])->GetCurrState() == LIVE)
        {
            countNeighbors++;
        }
    }
    bool alive = (GetCurrState() == LIVE);
    if ((alive && ((countNeighbors == 2) || (countNeighbors == 3))) || (!alive && (countNeighbors == 3)))
    {
        pGrid->AddLive();
        SetNextState(LIVE);
    }
    else
    {
        SetNextState(DEAD);
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t xDim = 200;
        const int32_t yDim = 200;
        const double  livingProb = 0.35;
        const int32_t runTicks = headless ? 100 : 10000000;

        Hal::Rand rng;
        GolGrid gol(xDim, yDim, livingProb, rng);
        Hal::Examples::GridWindow win("Game of Life", xDim, yDim, 3, !headless);

        for (int32_t i = 0; (i < runTicks) && !win.IsClosed(); i++)
        {
            gol.StepAgents();
            for (int32_t j = 0; j < gol.Length(); j++)
            {
                win.SetPix(j, gol.GetAgent(j)->GetCurrState());
            }
            win.TickPause(0);
        }

        if (headless)
        {
            win.ToPng("gameoflife.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GameOfLife failed: ") + e.what());
        return 1;
    }
}
