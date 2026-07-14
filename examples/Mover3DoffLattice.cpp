// Ported from LEARN_HERE/Agents/mover3DoffLattice: a single off-lattice
// (point) agent random-walking in continuous 3D space, drawn as a red sphere
// in a blue wireframe box via the software Scene3D renderer.
//
// Live window by default (drag to orbit, up/down to zoom); pass "headless"
// to run a bounded number of steps and dump the final frame to
// mover3dofflattice.png.

#include "Hal/Agents/AgentPT3D.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid3D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"
#include "common/Scene3D.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

class Mover3DoffLattice;

class MoveAgentOffLattice : public Hal::AgentPT3D<Mover3DoffLattice>
{
public:
    explicit MoveAgentOffLattice(Mover3DoffLattice *pGrid)
        : Hal::AgentPT3D<Mover3DoffLattice>(pGrid)
    {
    }
};

class Mover3DoffLattice : public Hal::AgentGrid3D<MoveAgentOffLattice, Mover3DoffLattice>
{
public:
    Mover3DoffLattice(int32_t x,
                      int32_t y,
                      int32_t z)
        : Hal::AgentGrid3D<MoveAgentOffLattice, Mover3DoffLattice>(x, y, z, false, false, false)
    {
    }
};

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        Mover3DoffLattice test(10, 10, 10);
        Hal::Examples::GridWindow win("3D", 500, 500, 1, !headless);
        Hal::Examples::Scene3D    scene(win, test.Xdim(), test.Ydim(), test.Zdim());

        MoveAgentOffLattice &hero = test.NewAgentSQ(5, 5, 5);
        Hal::Rand            rng;
        double               moveScratch[3] = {0.0, 0.0, 0.0};

        int32_t steps = headless ? 300 : 10000;
        for (int32_t i = 0; (i < steps) && !win.IsClosed(); i++)
        {
            rng.RandomPointInSphere(0.4, moveScratch);
            hero.MoveSafePT(hero.Xpt() + moveScratch[0], hero.Ypt() + moveScratch[1], hero.Zpt() + moveScratch[2]);

            scene.HandleInput();
            scene.ClearBox(Hal::BLUE, Hal::BLACK);
            scene.Sphere(hero.Xpt(), hero.Ypt(), hero.Zpt(), 0.5, Hal::RED);
            scene.Render();
            if (!headless)
            {
                win.TickPause(10);
            }
        }

        if (headless)
        {
            win.ToPng("mover3dofflattice.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("Mover3DoffLattice failed: ") + e.what());
        return 1;
    }
}
