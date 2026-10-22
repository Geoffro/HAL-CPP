// A growing tumor whose cells divide into empty von Neumann neighbors,
// die, and occasionally mutate. Each mutation recolors the cell
// (CategoricalColor) and multiplies its division rate, so fitter clones
// sweep outward.
//
// Live window by default; pass "headless" to dump the final state to
// divisiondeathmutation.png.

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

class DivisionDeathMutation;

class CellEx : public Hal::AgentSQ2DUnstackable<DivisionDeathMutation>
{
public:
    explicit CellEx(DivisionDeathMutation *pGrid)
        : Hal::AgentSQ2DUnstackable<DivisionDeathMutation>(pGrid),
          m_numMutations(0)
    {
    }

    void Mutate();
    void Draw();
    void Divide();

    int32_t NumMutations() const { return m_numMutations; }
    void    SetNumMutations(int32_t numMutations) { m_numMutations = numMutations; }

private:
    int32_t m_numMutations;
};

class DivisionDeathMutation : public Hal::AgentGrid2D<CellEx, DivisionDeathMutation>
{
public:
    DivisionDeathMutation(int32_t                      xDim,
                          int32_t                      yDim,
                          Hal::Examples::GridWindow   &vis)
        : Hal::AgentGrid2D<CellEx, DivisionDeathMutation>(xDim, yDim),
          m_hood(Hal::VonNeumannHood(false)),
          m_rng(1),
          m_vis(vis),
          m_mutCounts(MAX_MUTATIONS + 1, 0)
    {
    }

    void InitTumor(double radius)
    {
        std::vector<int32_t> circleHood = Hal::CircleHood(true, radius);
        int32_t len = MapHood(circleHood.data(), static_cast<int32_t>(circleHood.size()), Xdim() / 2, Ydim() / 2);
        for (int32_t i = 0; i < len; i++)
        {
            CellEx &c = NewAgentSQ(circleHood[i]);
            c.SetNumMutations(0);
            c.Draw();
        }
    }

    void StepCells()
    {
        std::fill(m_mutCounts.begin(), m_mutCounts.end(), 0);
        for (CellEx &c : *this)
        {
            m_mutCounts[c.NumMutations()]++;
            if (m_rng.Double() < DIE_PROB)
            {
                m_vis.SetPix(c.Isq(), BLACK);
                c.Dispose();
            }
            else if (m_rng.Double() < (DIV_PROB * std::pow(MUT_ADVANTAGE, c.NumMutations())))
            {
                c.Divide();
            }
        }
        ShuffleAgents(m_rng);
    }

    Hal::Examples::GridWindow &Vis() { return m_vis; }
    Hal::Rand                 &Rng() { return m_rng; }
    int32_t                   *HoodData() { return m_hood.data(); }
    int32_t                    HoodSize() const { return static_cast<int32_t>(m_hood.size()); }

    static const int32_t    BLACK;
    static constexpr double DIV_PROB = 0.2;
    static constexpr double MUT_PROB = 0.003;
    static constexpr double DIE_PROB = 0.1;
    static constexpr double MUT_ADVANTAGE = 1.08;
    static constexpr int32_t MAX_MUTATIONS = 19;

private:
    std::vector<int32_t>       m_hood;
    Hal::Rand                  m_rng;
    Hal::Examples::GridWindow &m_vis;
    std::vector<int32_t>       m_mutCounts;
};

const int32_t DivisionDeathMutation::BLACK = Hal::BLACK;

void CellEx::Mutate()
{
    DivisionDeathMutation *pGrid = this->Grid();
    if ((m_numMutations < DivisionDeathMutation::MAX_MUTATIONS) && (pGrid->Rng().Double() < DivisionDeathMutation::MUT_PROB))
    {
        m_numMutations++;
        Draw();
    }
}

void CellEx::Draw()
{
    this->Grid()->Vis().SetPix(Isq(), Hal::CategoricalColor(m_numMutations));
}

void CellEx::Divide()
{
    DivisionDeathMutation *pGrid = this->Grid();
    int32_t                nOpts = pGrid->MapEmptyHood(pGrid->HoodData(), pGrid->HoodSize(), Isq());
    if (nOpts > 0)
    {
        int32_t iDaughter = pGrid->HoodData()[pGrid->Rng().Int(nOpts)];
        CellEx &daughter = pGrid->NewAgentSQ(iDaughter);
        daughter.SetNumMutations(m_numMutations);
        daughter.Draw();
        Mutate();
        daughter.Mutate();
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t x = 500;
        const int32_t y = 500;
        Hal::Examples::GridWindow vis("Division / Death / Mutation", x, y, headless ? 1 : 2, !headless);
        DivisionDeathMutation grid(x, y, vis);
        grid.InitTumor(5);

        for (int32_t tick = 0; (tick < 2000) && !vis.IsClosed(); tick++)
        {
            vis.TickPause(0);
            grid.StepCells();
        }

        if (headless)
        {
            vis.ToPng("divisiondeathmutation.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("DivisionDeathMutation failed: ") + e.what());
        return 1;
    }
}
