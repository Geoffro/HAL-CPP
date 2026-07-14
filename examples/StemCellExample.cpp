// Ported from Examples/_2StemCellExample: a stem-cell-driven tissue where
// stem cells divide indefinitely (probabilistically producing another stem
// cell) while non-stem cells have a finite division budget (divs, a
// telomere-shortening analogue) before they can no longer divide and die.
// Cells divide into empty Moore neighbors and die randomly.
//
// The Java version wraps this in an interactive parameter menu; this port
// uses that menu's default parameters directly. Stem cells draw red,
// non-stem cells shade blue by remaining division budget.
//
// Live window by default; pass "headless" to dump the final tissue to
// stemcellexample.png.

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

class StemCellCA;

class CACell : public Hal::AgentSQ2DUnstackable<StemCellCA>
{
public:
    explicit CACell(StemCellCA *pGrid)
        : Hal::AgentSQ2DUnstackable<StemCellCA>(pGrid),
          m_divs(0),
          m_stem(false)
    {
    }

    void Init(int32_t d, bool s)
    {
        m_divs = d;
        m_stem = s;
    }

    void Divide(int32_t iChildLoc);
    void Step();

    int32_t Divs() const { return m_divs; }
    bool    Stem() const { return m_stem; }

private:
    int32_t m_divs;
    bool    m_stem;
};

class StemCellCA : public Hal::AgentGrid2D<CACell, StemCellCA>
{
public:
    StemCellCA(int32_t xDim,
               int32_t yDim,
               Hal::Examples::GridWindow &vis)
        : Hal::AgentGrid2D<CACell, StemCellCA>(xDim, yDim, true, true),
          m_mooreHood(Hal::MooreHood(false)),
          m_vis(vis)
    {
        vis.Clear(Hal::BLACK);
    }

    void Draw()
    {
        for (int32_t i = 0; i < Length(); i++)
        {
            CACell *pCell = GetAgent(i);
            if (pCell == nullptr)
            {
                m_vis.SetPix(i, Hal::BLACK);
            }
            else if (pCell->Stem())
            {
                m_vis.SetPix(i, Hal::RED);
            }
            else
            {
                m_vis.SetPix(i, Hal::Rgb(0.0, 0.0, (pCell->Divs() + 1.0) / MAX_DIVS));
            }
        }
    }

    void Run(int32_t runDuration)
    {
        for (int32_t i = 0; (i < runDuration) && !m_vis.IsClosed(); i++)
        {
            if (Pop() == 0)
            {
                NewAgentSQ(Xdim() / 2, Ydim() / 2).Init(MAX_DIVS, true);
            }
            for (CACell &c : *this)
            {
                c.Step();
            }
            Draw();
            CleanAgents();
            ShuffleAgents(m_rng);
            m_vis.TickPause(TICK_PAUSE);
        }
    }

    Hal::Rand &Rng() { return m_rng; }
    int32_t   *MooreHoodData() { return m_mooreHood.data(); }
    int32_t    MooreHoodSize() const { return static_cast<int32_t>(m_mooreHood.size()); }

    static constexpr double DIV_PROB = 1.0 / 24;
    static constexpr double DEATH_PROB = 1.0 / 1000;
    static constexpr double STEM_DIV_PROB = 7.0 / 10;
    static constexpr int32_t MAX_DIVS = 11;
    static constexpr int32_t TICK_PAUSE = 0;

private:
    std::vector<int32_t>       m_mooreHood;
    Hal::Examples::GridWindow &m_vis;
    Hal::Rand                  m_rng;
};

void CACell::Divide(int32_t iChildLoc)
{
    StemCellCA *pGrid = this->Grid();
    bool        stemChild = false;
    int32_t     divsChild = m_divs;
    if (m_stem && (pGrid->Rng().Double() < StemCellCA::STEM_DIV_PROB))
    {
        stemChild = true;
    }
    else
    {
        divsChild--;
    }
    pGrid->NewAgentSQ(iChildLoc).Init(divsChild, stemChild);
    if (!m_stem)
    {
        m_divs--;
    }
}

void CACell::Step()
{
    StemCellCA *pGrid = this->Grid();
    if (pGrid->Rng().Double() < StemCellCA::DEATH_PROB)
    {
        Dispose();
        return;
    }
    if (pGrid->Rng().Double() < StemCellCA::DIV_PROB)
    {
        int32_t ct = pGrid->MapEmptyHood(pGrid->MooreHoodData(), pGrid->MooreHoodSize(), Isq());
        if (ct > 0)
        {
            if (m_divs == 0)
            {
                Dispose();
                return;
            }
            Divide(pGrid->MooreHoodData()[pGrid->Rng().Int(ct)]);
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t dim = 200;
        Hal::Examples::GridWindow vis("Stem Cell CA", dim, dim, headless ? 2 : 5, !headless);
        StemCellCA grid(dim, dim, vis);

        grid.Run(headless ? 2000 : 20000);

        if (headless)
        {
            vis.ToPng("stemcellexample.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("StemCellExample failed: ") + e.what());
        return 1;
    }
}
