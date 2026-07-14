// Ported from Examples/GasPopulationGrid/Gas: many off-lattice particles
// undergoing Brownian motion, visualized as a per-cell density heat map
// (PopAt counts how many particles share a square). Starting as a point
// mass at the center, the gas diffuses to fill the domain.
//
// Live window by default; pass "headless" to run fewer steps and dump the
// final density to gas.png.

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

class Gas;

class Particle : public Hal::AgentPT2D<Gas>
{
public:
    explicit Particle(Gas *pGrid)
        : Hal::AgentPT2D<Gas>(pGrid)
    {
    }

    void BrownianMotion();
};

class Gas : public Hal::AgentGrid2D<Particle, Gas>
{
public:
    Gas(int32_t x,
        int32_t y)
        : Hal::AgentGrid2D<Particle, Gas>(x, y)
    {
    }

    void Draw(Hal::Examples::GridWindow &display)
    {
        for (int32_t i = 0; i < Length(); i++)
        {
            // 0 particles -> black, 4+ -> white.
            display.SetPix(i, Hal::HeatMapRgb(PopAt(i), 0, 4));
        }
    }

    Hal::Rand &Rng() { return m_rng; }
    double    *MoveCoords() { return m_moveCoords; }

private:
    Hal::Rand m_rng;
    double    m_moveCoords[2] = {0.0, 0.0};
};

void Particle::BrownianMotion()
{
    Gas *pGrid = this->Grid();
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
        Hal::Examples::GridWindow display("Gas density", x, y, 5, !headless);
        Gas gas(x, y);

        for (int32_t i = 0; i < 10000; i++)
        {
            gas.NewAgentPT(x / 2.0, y / 2.0);
        }

        int32_t steps = headless ? 2000 : 100000;
        for (int32_t i = 0; (i < steps) && !display.IsClosed(); i++)
        {
            display.TickPause(0);
            for (Particle &particle : gas)
            {
                particle.BrownianMotion();
            }
            gas.Draw(display);
        }

        if (headless)
        {
            display.ToPng("gas.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("Gas failed: ") + e.what());
        return 1;
    }
}
