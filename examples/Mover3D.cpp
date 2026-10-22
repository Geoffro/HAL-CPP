// A single agent random-walking on a 3D lattice, shown in a software 3D
// window (Scene3D) as a green sphere inside a magenta wireframe box,
// alongside a 2D top-down view colored by height.
//
// Live windows by default; pass "headless" to run a bounded number of steps
// and dump the final 3D frame to mover3d.png. Drag the left mouse to orbit;
// up/down arrows zoom.

#include "Hal/Agents/AgentSQ3DUnstackable.h"
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

class Mover3D;

class MoveAgent : public Hal::AgentSQ3DUnstackable<Mover3D>
{
public:
    explicit MoveAgent(Mover3D *pGrid)
        : Hal::AgentSQ3DUnstackable<Mover3D>(pGrid)
    {
    }
};

class Mover3D : public Hal::AgentGrid3D<MoveAgent, Mover3D>
{
public:
    Mover3D(int32_t x,
            int32_t y,
            int32_t z)
        : Hal::AgentGrid3D<MoveAgent, Mover3D>(x, y, z, false, false, false)
    {
    }
};

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        Mover3D test(10, 10, 10);
        Hal::Examples::GridWindow win3D("3D", 500, 500, 1, !headless);
        Hal::Examples::Scene3D    scene(win3D, test.Xdim(), test.Ydim(), test.Zdim());
        Hal::Examples::GridWindow win2D("2D", test.Xdim(), test.Ydim(), 20, !headless);

        MoveAgent &hero = test.NewAgentSQ(5, 5, 5);
        Hal::Rand  rng;

        int32_t steps = headless ? 300 : 10000;
        for (int32_t i = 0; (i < steps) && !win3D.IsClosed() && !win2D.IsClosed(); i++)
        {
            hero.MoveSafeSQ(hero.Xsq() + (rng.Int(3) - 1),
                            hero.Ysq() + (rng.Int(3) - 1),
                            hero.Zsq() + (rng.Int(3) - 1));

            win2D.Clear(Hal::BLACK);
            win2D.SetPix(hero.Xsq(), hero.Ysq(), Hal::HeatMapRgb(hero.Zsq(), 0, test.Zdim()));
            win2D.TickPause(headless ? 0 : 10);

            scene.HandleInput();
            scene.ClearBox(Hal::MAGENTA, Hal::BLACK);
            scene.Sphere(hero.Xsq(), hero.Ysq(), hero.Zsq(), 0.5, Hal::GREEN);
            scene.Render();
        }

        if (headless)
        {
            win3D.ToPng("mover3d.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("Mover3D failed: ") + e.what());
        return 1;
    }
}
