// Minimal off-lattice agent skeleton: a single SphericalAgent2D that
// recolors, resizes, and jitters its velocity each step, moving under its
// own momentum with friction. A starting point to copy from.
//
// Live window by default; pass "headless" to run a bounded number of steps
// and dump the final frame to templateofflattice2d.png.

#include "Hal/Agents/SphericalAgent2D.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

class TemplateOffLattice2D;

class OffLatticeCell : public Hal::SphericalAgent2D<TemplateOffLattice2D>
{
public:
    explicit OffLatticeCell(TemplateOffLattice2D *pGrid)
        : Hal::SphericalAgent2D<TemplateOffLattice2D>(pGrid),
          m_exampleProperty(0)
    {
    }

    void ExampleFunction();

    int32_t ExampleProperty() const { return m_exampleProperty; }

private:
    int32_t m_exampleProperty;
};

class TemplateOffLattice2D : public Hal::AgentGrid2D<OffLatticeCell, TemplateOffLattice2D>
{
public:
    TemplateOffLattice2D(int32_t x,
                         int32_t y)
        : Hal::AgentGrid2D<OffLatticeCell, TemplateOffLattice2D>(x, y)
    {
    }

    Hal::Rand &Rng() { return m_rng; }

private:
    Hal::Rand m_rng;
};

void OffLatticeCell::ExampleFunction()
{
    TemplateOffLattice2D *pGrid = this->Grid();
    m_exampleProperty = Hal::Rgb(pGrid->Rng().Double(), pGrid->Rng().Double(), pGrid->Rng().Double());
    SetRadius(pGrid->Rng().Double());
    SetVelocity(XVel() + pGrid->Rng().Double(2) - 1, YVel() + pGrid->Rng().Double(2) - 1);
    ForceMove();
    ApplyFriction(0.0);
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        TemplateOffLattice2D grid(11, 11);
        Hal::Examples::GridWindow vis("Off-Lattice Template", 11, 11, 45, !headless);

        grid.NewAgentPT(grid.Xdim() / 2.0, grid.Ydim() / 2.0);

        int32_t maxSteps = headless ? 50 : 1000000;
        for (int32_t step = 0; (step < maxSteps) && !vis.IsClosed(); step++)
        {
            for (OffLatticeCell &cell : grid)
            {
                cell.ExampleFunction();
            }
            vis.Clear(Hal::BLACK);
            for (OffLatticeCell &cell : grid)
            {
                vis.Circle(cell.Xpt(), cell.Ypt(), cell.Radius(), cell.ExampleProperty());
            }
            vis.TickPause(headless ? 0 : 1000);
        }

        if (headless)
        {
            vis.ToPng("templateofflattice2d.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("TemplateOffLattice2D failed: ") + e.what());
        return 1;
    }
}
