// Ported from Examples/Template_Models/TemplateOnLattice2D: the minimal
// on-lattice agent skeleton — a single cell random-walks its von-Neumann
// neighborhood, recoloring itself each step. Meant as a starting point to
// copy from, so it stays deliberately tiny.
//
// (Java uses the stackable AgentSQ2D; this port uses the unstackable base,
// which is equivalent here since there is only ever one agent.)
//
// Live window by default; pass "headless" to run a bounded number of steps
// and dump the final frame to templateonlattice2d.png.

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

class TemplateOnLattice2D;

class TemplateCell : public Hal::AgentSQ2DUnstackable<TemplateOnLattice2D>
{
public:
    explicit TemplateCell(TemplateOnLattice2D *pGrid)
        : Hal::AgentSQ2DUnstackable<TemplateOnLattice2D>(pGrid),
          m_exampleProperty(0)
    {
    }

    void ExampleFunction();

    int32_t ExampleProperty() const { return m_exampleProperty; }

private:
    int32_t m_exampleProperty;
};

class TemplateOnLattice2D : public Hal::AgentGrid2D<TemplateCell, TemplateOnLattice2D>
{
public:
    TemplateOnLattice2D(int32_t xDim,
                        int32_t yDim)
        : Hal::AgentGrid2D<TemplateCell, TemplateOnLattice2D>(xDim, yDim),
          m_hood(Hal::VonNeumannHood(false))
    {
    }

    Hal::Rand &Rng() { return m_rng; }
    int32_t   *HoodData() { return m_hood.data(); }
    int32_t    HoodSize() const { return static_cast<int32_t>(m_hood.size()); }

private:
    Hal::Rand            m_rng;
    std::vector<int32_t> m_hood;
};

void TemplateCell::ExampleFunction()
{
    TemplateOnLattice2D *pGrid = this->Grid();
    m_exampleProperty = Hal::Rgb(pGrid->Rng().Double(), pGrid->Rng().Double(), pGrid->Rng().Double());
    int32_t nOpts = MapHood(pGrid->HoodData(), pGrid->HoodSize());
    MoveSQ(pGrid->HoodData()[pGrid->Rng().Int(nOpts)]);
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        TemplateOnLattice2D grid(11, 11);
        Hal::Examples::GridWindow vis("On-Lattice Template", 11, 11, 50, !headless);

        grid.NewAgentSQ(grid.Xdim() / 2, grid.Ydim() / 2);

        int32_t maxSteps = headless ? 50 : 1000000;
        for (int32_t step = 0; (step < maxSteps) && !vis.IsClosed(); step++)
        {
            for (TemplateCell &cell : grid)
            {
                cell.ExampleFunction();
            }
            for (int32_t i = 0; i < grid.Length(); i++)
            {
                TemplateCell *pCell = grid.GetAgent(i);
                vis.SetPix(i, pCell != nullptr ? pCell->ExampleProperty() : Hal::BLACK);
            }
            vis.TickPause(headless ? 0 : 1000);
        }

        if (headless)
        {
            vis.ToPng("templateonlattice2d.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("TemplateOnLattice2D failed: ") + e.what());
        return 1;
    }
}
