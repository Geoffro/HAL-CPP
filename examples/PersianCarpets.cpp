// A spatial cooperator vs. defector evolutionary game. Each cell's fitness
// is the payoff summed over its Moore neighborhood; every cell then
// synchronously adopts the type of its fittest neighbor. A single central
// defector grows the self-similar "Persian carpet" patterns. Cells are
// colored by their (previous type -> next type) transition.
//
// Live window by default; pass "headless" to dump the final carpet to
// persiancarpets.png.

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Neighborhood.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

class PersianCarpets;

class CarpetCell : public Hal::AgentSQ2DUnstackable<PersianCarpets>
{
public:
    static constexpr int32_t COOPERATOR = 0;
    static constexpr int32_t DEFECTOR = 1;

    explicit CarpetCell(PersianCarpets *pGrid)
        : Hal::AgentSQ2DUnstackable<PersianCarpets>(pGrid),
          m_type(COOPERATOR),
          m_futureType(COOPERATOR),
          m_fitness(0.0)
    {
    }

    void DetermineFitness();
    void SetFutureType();

    int32_t Type() const { return m_type; }
    void    SetType(int32_t type) { m_type = type; }
    int32_t FutureType() const { return m_futureType; }
    double  Fitness() const { return m_fitness; }

private:
    int32_t m_type;
    int32_t m_futureType;
    double  m_fitness;
};

class PersianCarpets : public Hal::AgentGrid2D<CarpetCell, PersianCarpets>
{
public:
    static constexpr int32_t DIMENSION = 101;
    static constexpr int32_t SCALE_FACTOR = 5;
    static constexpr int32_t PAUSE = 50;

    explicit PersianCarpets(Hal::Examples::GridWindow &vis)
        : Hal::AgentGrid2D<CarpetCell, PersianCarpets>(DIMENSION, DIMENSION, true, true),
          m_hood(Hal::MooreHood(false)),
          m_vis(vis)
    {
        const double b = 1.61; // try 1.35 or 1.61
        m_payoff = {1.0, 0.0, b, 0.0};

        for (int32_t i = 0; i < Length(); i++)
        {
            NewAgentSQ(i).SetType(CarpetCell::COOPERATOR);
        }
        GetAgent((Xdim() - 1) / 2, (Xdim() - 1) / 2)->SetType(CarpetCell::DEFECTOR);
    }

    void StepCells()
    {
        for (CarpetCell &c : *this)
        {
            c.DetermineFitness();
        }
        for (CarpetCell &c : *this)
        {
            c.SetFutureType();
        }
        Draw();
        for (CarpetCell &c : *this)
        {
            c.SetType(c.FutureType());
        }
    }

    void Draw()
    {
        for (int32_t i = 0; i < Length(); i++)
        {
            CarpetCell *pCell = GetAgent(i);
            if (pCell == nullptr)
            {
                continue;
            }
            bool defNow = (pCell->Type() == CarpetCell::DEFECTOR);
            bool defNext = (pCell->FutureType() == CarpetCell::DEFECTOR);
            if (defNow && defNext)
            {
                m_vis.SetPix(i, Hal::RED);
            }
            else if (defNow && !defNext)
            {
                m_vis.SetPix(i, Hal::GREEN);
            }
            else if (!defNow && defNext)
            {
                m_vis.SetPix(i, Hal::YELLOW);
            }
            else
            {
                m_vis.SetPix(i, Hal::BLUE);
            }
        }
        m_vis.TickPause(m_vis.IsClosed() ? 0 : PAUSE);
    }

    const std::vector<double> &Payoff() const { return m_payoff; }
    int32_t                   *HoodData() { return m_hood.data(); }
    int32_t                    HoodSize() const { return static_cast<int32_t>(m_hood.size()); }

private:
    std::vector<double>        m_payoff;
    std::vector<int32_t>       m_hood;
    Hal::Examples::GridWindow &m_vis;
};

void CarpetCell::DetermineFitness()
{
    PersianCarpets *pGrid = this->Grid();
    int32_t         nNeighbors = MapHood(pGrid->HoodData(), pGrid->HoodSize());
    int32_t         neighborsTypes[2] = {0, 0};
    for (int32_t i = 0; i < nNeighbors; i++)
    {
        neighborsTypes[pGrid->GetAgent(pGrid->HoodData()[i])->Type()]++;
    }
    const std::vector<double> &payoff = pGrid->Payoff();
    if (m_type == COOPERATOR)
    {
        m_fitness = payoff[0] * neighborsTypes[0] + payoff[1] * neighborsTypes[1];
    }
    else
    {
        m_fitness = payoff[2] * neighborsTypes[0] + payoff[3] * neighborsTypes[1];
    }
}

void CarpetCell::SetFutureType()
{
    PersianCarpets *pGrid = this->Grid();
    double          maxFitness = 0.0; // assumes all payoff entries are non-negative
    int32_t         chosen = m_type;
    int32_t         nNeighbors = MapHood(pGrid->HoodData(), pGrid->HoodSize());
    for (int32_t i = 0; i < nNeighbors; i++)
    {
        CarpetCell *pNeighbor = pGrid->GetAgent(pGrid->HoodData()[i]);
        if (pNeighbor->Fitness() > maxFitness)
        {
            maxFitness = pNeighbor->Fitness();
            chosen = pNeighbor->Type();
        }
    }
    m_futureType = chosen;
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        Hal::Examples::GridWindow vis("Persian Carpets", PersianCarpets::DIMENSION, PersianCarpets::DIMENSION,
                                      PersianCarpets::SCALE_FACTOR, !headless);
        PersianCarpets grid(vis);

        grid.Draw();
        for (int32_t tick = 0; (tick < 200) && !vis.IsClosed(); tick++)
        {
            grid.StepCells();
            grid.Draw();
        }

        if (headless)
        {
            vis.ToPng("persiancarpets.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("PersianCarpets failed: ") + e.what());
        return 1;
    }
}
