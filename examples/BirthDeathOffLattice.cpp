// Ported from LEARN_HERE/Agents/BirthDeathOffLattice: an off-lattice
// (point) birth/death model. Cells live at continuous positions, die
// randomly, divide (capped at 5 per square), and random-walk a little each
// step. Java draws them with OpenGL2DWindow.Circle; here GridWindow.Circle
// rasterizes each cell as a small disc.
//
// Live window by default; pass "headless" to run a bounded number of steps
// and dump the final frame to birthdeathofflattice.png.

#include "Hal/Agents/AgentPT2D.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

class BirthDeathOffLattice;

class OffLatticeCell : public Hal::AgentPT2D<BirthDeathOffLattice>
{
public:
    explicit OffLatticeCell(BirthDeathOffLattice *pGrid)
        : Hal::AgentPT2D<BirthDeathOffLattice>(pGrid),
          m_color(0)
    {
    }

    void Init();
    void StepCell(double dieProb, double divProb);

    int32_t Color() const { return m_color; }

private:
    int32_t m_color;
};

class BirthDeathOffLattice : public Hal::AgentGrid2D<OffLatticeCell, BirthDeathOffLattice>
{
public:
    BirthDeathOffLattice(int32_t x,
                         int32_t y)
        : Hal::AgentGrid2D<OffLatticeCell, BirthDeathOffLattice>(x, y)
    {
    }

    void Step(double dieProb,
              double divProb)
    {
        for (OffLatticeCell &cell : *this)
        {
            cell.StepCell(dieProb, divProb);
        }
    }

    void DrawModel(Hal::Examples::GridWindow &win)
    {
        win.Clear(Hal::BLACK);
        for (OffLatticeCell &cell : *this)
        {
            win.Circle(cell.Xpt(), cell.Ypt(), 0.5, cell.Color());
        }
        win.Tick();
    }

    Hal::Rand &Rng() { return m_rng; }
    double    *MoveCoords() { return m_moveCoords; }

private:
    Hal::Rand m_rng;
    double    m_moveCoords[2] = {0.0, 0.0};
};

void OffLatticeCell::Init()
{
    BirthDeathOffLattice *pGrid = this->Grid();
    m_color = Hal::Rgb(pGrid->Rng().Double(), pGrid->Rng().Double(), pGrid->Rng().Double());
}

void OffLatticeCell::StepCell(double dieProb,
                              double divProb)
{
    BirthDeathOffLattice *pGrid = this->Grid();
    if (pGrid->Rng().Double() < dieProb)
    {
        Dispose();
        return;
    }
    if ((pGrid->Rng().Double() < divProb) && (pGrid->PopAt(Isq()) < 5))
    {
        pGrid->NewAgentPT(Xpt(), Ypt()).Init();
    }
    pGrid->Rng().RandomPointInCircle(0.5, pGrid->MoveCoords());
    MoveSafePT(Xpt() + pGrid->MoveCoords()[0], Ypt() + pGrid->MoveCoords()[1]);
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t x = 100;
        const int32_t y = 100;
        const double  dieProb = 0.1;
        const double  divProb = 0.2;

        Hal::Examples::GridWindow win("Off-Lattice Birth/Death", x, y, 5, !headless);
        BirthDeathOffLattice model(x, y);

        int32_t timesteps = headless ? 300 : 1000;
        for (int32_t i = 0; (i < timesteps) && !win.IsClosed(); i++)
        {
            if (model.Pop() == 0)
            {
                model.NewAgentPT(x / 2.0, y / 2.0).Init();
            }
            model.Step(dieProb, divProb);
            model.DrawModel(win);
        }

        if (headless)
        {
            win.ToPng("birthdeathofflattice.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("BirthDeathOffLattice failed: ") + e.what());
        return 1;
    }
}
