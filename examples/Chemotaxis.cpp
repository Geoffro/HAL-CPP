// Off-lattice "cows" graze a regrowing grass PDE field and climb its
// gradient (chemotaxis) toward denser grass, while jostling each other via
// spherical contact forces. Grass is drawn green; cows are white discs.
//
// Live window by default; pass "headless" to run fewer steps and dump the
// final frame to chemotaxis.png.

#include "Hal/Agents/SphericalAgent2D.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Grids/PDEGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/MathUtil.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

class ChemotaxisModel;

class SphericalCow : public Hal::SphericalAgent2D<ChemotaxisModel>
{
public:
    explicit SphericalCow(ChemotaxisModel *pGrid)
        : Hal::SphericalAgent2D<ChemotaxisModel>(pGrid)
    {
    }

    void Step();
};

class ChemotaxisModel : public Hal::AgentGrid2D<SphericalCow, ChemotaxisModel>
{
public:
    ChemotaxisModel(int32_t x,
                    int32_t y)
        : Hal::AgentGrid2D<SphericalCow, ChemotaxisModel>(x, y),
          m_grass(x, y)
    {
        m_grass.SetAll(1);
        m_grass.Update();
    }

    void GrowGrass()
    {
        for (int32_t i = 0; i < m_grass.Length(); i++)
        {
            m_grass.Add(i, GROW_RATE);
        }
        m_grass.Update();
        for (int32_t i = 0; i < m_grass.Length(); i++)
        {
            if (m_grass.Get(i) > 1)
            {
                m_grass.Set(i, 1);
            }
        }
        m_grass.Update();
    }

    void Draw(Hal::Examples::GridWindow &win)
    {
        for (int32_t x = 0; x < Xdim(); x++)
        {
            for (int32_t y = 0; y < Ydim(); y++)
            {
                win.SetPix(x, y, Hal::Rgb(0.0, m_grass.Get(x, y) / 2, 0.0));
            }
        }
        for (SphericalCow &cow : *this)
        {
            win.Circle(cow.Xpt(), cow.Ypt(), cow.Radius(), Hal::WHITE);
        }
        win.Tick();
    }

    Hal::PDEGrid2D &Grass() { return m_grass; }
    Hal::Rand      &Rng() { return m_rng; }
    double         *RandomMoveScratch() { return m_randomMoveScratch; }

    static constexpr double FORCE_EXPONENT = 2;
    static constexpr double FORCE_SCALER = 0.7;
    static constexpr double FRICTION = 0.5;
    static constexpr double GROW_RATE = 0.01;
    static constexpr double EAT_RATE = 1;
    static constexpr double CHEMOTAX_RATE = 0.1;
    static constexpr double RANDOM_MOVE_RATE = 0.1;
    static constexpr double COW_RAD = 0.5;

private:
    Hal::PDEGrid2D m_grass;
    Hal::Rand      m_rng;
    double         m_randomMoveScratch[2] = {0.0, 0.0};
};

void SphericalCow::Step()
{
    ChemotaxisModel *pGrid = this->Grid();
    if (pGrid->Grass().Get(Isq()) > 0)
    {
        pGrid->Grass().Add(Isq(), -ChemotaxisModel::EAT_RATE);
    }
    // Chemotaxis: climb the grass gradient.
    double gradX = pGrid->Grass().GradientX(Xsq(), Ysq());
    double gradY = pGrid->Grass().GradientY(Xsq(), Ysq());
    double norm = Hal::Norm(gradX, gradY);
    double newXVel = XVel();
    double newYVel = YVel();
    if (gradX != 0)
    {
        newXVel += gradX / norm * ChemotaxisModel::CHEMOTAX_RATE;
    }
    if (gradY != 0)
    {
        newYVel += gradY / norm * ChemotaxisModel::CHEMOTAX_RATE;
    }
    // Random movement.
    pGrid->Rng().RandomPointInCircle(ChemotaxisModel::RANDOM_MOVE_RATE, pGrid->RandomMoveScratch());
    newXVel += pGrid->RandomMoveScratch()[0];
    newYVel += pGrid->RandomMoveScratch()[1];
    SetVelocity(newXVel, newYVel);
    // Contact forces with nearby cows.
    SumForces(Radius(), [](double overlap, SphericalCow &) {
        return std::pow(ChemotaxisModel::FORCE_SCALER * overlap, ChemotaxisModel::FORCE_EXPONENT);
    });
    ForceMove();
    ApplyFriction(ChemotaxisModel::FRICTION);
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t x = 200;
        const int32_t y = 200;
        Hal::Examples::GridWindow win("Chemotaxis", x, y, headless ? 2 : 3, !headless);
        ChemotaxisModel m(x, y);
        for (int32_t i = 0; i < 500; i++)
        {
            m.NewAgentSQ(m.Rng().Int(m.Length())).SetRadius(ChemotaxisModel::COW_RAD);
        }

        int32_t steps = headless ? 300 : 1000000;
        for (int32_t step = 0; (step < steps) && !win.IsClosed(); step++)
        {
            for (SphericalCow &cow : m)
            {
                cow.Step();
            }
            m.GrowGrass();
            m.Draw(win);
            win.TickPause(0);
        }

        if (headless)
        {
            win.ToPng("chemotaxis.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("Chemotaxis failed: ") + e.what());
        return 1;
    }
}
