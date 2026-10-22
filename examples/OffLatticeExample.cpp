// Two competing cell types (purple / pink) as spherical off-lattice
// agents. Cells push apart via Hooke's-law contact forces and divide at a
// type-specific rate that is contact-inhibited by the total force they
// feel, so crowding slows growth. Cells draw as a colored nucleus on a
// shared "cytoplasm" disc. Writes per-type populations to poponts.csv.
//
// Live window by default; pass "headless" to run fewer steps and dump the
// final frame to offlatticeexample.png.

#include "Hal/Agents/SphericalAgent2D.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>

class ExampleOffLattice;

class CellOL : public Hal::SphericalAgent2D<ExampleOffLattice>
{
public:
    explicit CellOL(ExampleOffLattice *pGrid)
        : Hal::SphericalAgent2D<ExampleOffLattice>(pGrid),
          m_type(0),
          m_forceSum(0.0)
    {
    }

    void   Init(int32_t color);
    double ForceCalc(double overlap, CellOL &other);
    void   CalcMove();
    bool   CanDivide(double divBias, double inhibWeight);
    void   MoveDiv();

    int32_t Type() const { return m_type; }

private:
    int32_t m_type;
    double  m_forceSum;
};

class ExampleOffLattice : public Hal::AgentGrid2D<CellOL, ExampleOffLattice>
{
public:
    static const int32_t WHITE_BG;
    static const int32_t PURPLE;
    static const int32_t PINK;
    static const int32_t CYTOPLASM;

    ExampleOffLattice(int32_t x,
                      int32_t y)
        : Hal::AgentGrid2D<CellOL, ExampleOffLattice>(x, y, true, true),
          m_rng(0)
    {
    }

    void Setup(int32_t initPop,
               double  initRadius,
               double  propRed)
    {
        for (int32_t i = 0; i < initPop; i++)
        {
            m_rng.RandomPointInCircle(initRadius, m_divCoords);
            NewAgentPT(m_divCoords[0] + Xdim() / 2.0, m_divCoords[1] + Ydim() / 2.0)
                .Init(m_rng.Double() < propRed ? PURPLE : PINK);
        }
    }

    void StepCells(std::ofstream *pOut)
    {
        for (CellOL &cell : *this)
        {
            cell.CalcMove();
        }
        for (CellOL &cell : *this)
        {
            cell.MoveDiv();
        }
        if (pOut != nullptr)
        {
            int32_t ctPurp = 0;
            int32_t ctPink = 0;
            for (CellOL &cell : *this)
            {
                if (cell.Type() == PURPLE)
                {
                    ctPurp++;
                }
                else
                {
                    ctPink++;
                }
            }
            *pOut << ctPink << "," << ctPurp << "\n";
        }
    }

    void DrawCells(Hal::Examples::GridWindow &vis)
    {
        vis.Clear(WHITE_BG);
        for (CellOL &cell : *this)
        {
            vis.Circle(cell.Xpt(), cell.Ypt(), cell.Radius(), CYTOPLASM);
        }
        for (CellOL &cell : *this)
        {
            vis.Circle(cell.Xpt(), cell.Ypt(), cell.Radius() / 3, cell.Type());
        }
        vis.Tick();
    }

    Hal::Rand &Rng() { return m_rng; }
    double    *DivCoords() { return m_divCoords; }

    static constexpr double RADIUS = 0.25;
    static constexpr double FORCE_SCALER = 0.25;
    static constexpr double FRICTION = 0.5;
    static constexpr double PURP_DIV_BIAS = 0.01;
    static constexpr double PINK_DIV_BIAS = 0.02;
    static constexpr double PURP_INHIB_WEIGHT = 0.02;
    static constexpr double PINK_INHIB_WEIGHT = 0.05;

private:
    Hal::Rand m_rng;
    double    m_divCoords[2] = {0.0, 0.0};
};

const int32_t ExampleOffLattice::WHITE_BG = Hal::Rgb256(248, 255, 252);
const int32_t ExampleOffLattice::PURPLE = Hal::Rgb256(77, 0, 170);
const int32_t ExampleOffLattice::PINK = Hal::Rgb256(222, 0, 109);
const int32_t ExampleOffLattice::CYTOPLASM = Hal::Rgb256(191, 156, 147);

void CellOL::Init(int32_t color)
{
    m_type = color;
    SetRadius(ExampleOffLattice::RADIUS);
}

double CellOL::ForceCalc(double overlap, CellOL &)
{
    if (overlap < 0)
    {
        return 0; // not actually overlapping -> no force
    }
    return ExampleOffLattice::FORCE_SCALER * overlap; // Hooke's law
}

void CellOL::CalcMove()
{
    m_forceSum = SumForces(ExampleOffLattice::RADIUS * 2,
                         [this](double overlap, CellOL &other) { return ForceCalc(overlap, other); });
}

bool CellOL::CanDivide(double divBias, double inhibWeight)
{
    return this->Grid()->Rng().Double() < std::tanh(divBias - m_forceSum * inhibWeight);
}

void CellOL::MoveDiv()
{
    ExampleOffLattice *pGrid = this->Grid();
    ForceMove();
    ApplyFriction(ExampleOffLattice::FRICTION);
    if ((m_type == ExampleOffLattice::PURPLE && CanDivide(ExampleOffLattice::PURP_DIV_BIAS, ExampleOffLattice::PURP_INHIB_WEIGHT)) ||
        (m_type == ExampleOffLattice::PINK && CanDivide(ExampleOffLattice::PINK_DIV_BIAS, ExampleOffLattice::PINK_INHIB_WEIGHT)))
    {
        CellOL *pChild = Divide(Radius() * 2.0 / 3.0, pGrid->DivCoords(), &pGrid->Rng());
        if (pChild != nullptr)
        {
            pChild->Init(m_type);
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t x = 30;
        const int32_t y = 30;
        ExampleOffLattice ex(x, y);
        Hal::Examples::GridWindow vis("Off-Lattice Example", x, y, headless ? 10 : 33, !headless);
        std::ofstream out("poponts.csv");

        ex.Setup(50, 5, 0.5);

        int32_t steps = headless ? 500 : 1000;
        for (int32_t i = 0; (i < steps) && !vis.IsClosed(); i++)
        {
            vis.TickPause(0);
            ex.StepCells(&out);
            ex.DrawCells(vis);
        }
        out.close();

        if (headless)
        {
            vis.ToPng("offlatticeexample.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("OffLatticeExample failed: ") + e.what());
        return 1;
    }
}
