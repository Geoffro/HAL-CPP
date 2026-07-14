// Ported from Examples/Template_Models/TemplateMultiGrid: the multi-grid
// skeleton — an on-lattice agent grid, an off-lattice spherical agent grid,
// and a 3D PDE all sharing one domain, drawn together in the software 3D
// window. The PDE field is shown as a lattice of tiny colored spheres, with
// each agent as a larger sphere. A starting point to copy from.
//
// Live window by default (drag to orbit, up/down to zoom); pass "headless"
// to run a bounded number of steps and dump the final frame to
// templatemultigrid.png.

#include "Hal/Agents/AgentSQ3DUnstackable.h"
#include "Hal/Agents/SphericalAgent3D.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid3D.h"
#include "Hal/Grids/PDEGrid3D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"
#include "common/Scene3D.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

class Grid1;
class Grid2;

class CellType1 : public Hal::AgentSQ3DUnstackable<Grid1>
{
public:
    explicit CellType1(Grid1 *pGrid)
        : Hal::AgentSQ3DUnstackable<Grid1>(pGrid),
          m_exampleProperty(0)
    {
    }

    int32_t ExampleProperty() const { return m_exampleProperty; }
    void    SetExampleProperty(int32_t exampleProperty) { m_exampleProperty = exampleProperty; }

private:
    int32_t m_exampleProperty;
};

class Grid1 : public Hal::AgentGrid3D<CellType1, Grid1>
{
public:
    Grid1(int32_t x, int32_t y, int32_t z)
        : Hal::AgentGrid3D<CellType1, Grid1>(x, y, z)
    {
    }
};

class CellType2 : public Hal::SphericalAgent3D<Grid2>
{
public:
    explicit CellType2(Grid2 *pGrid)
        : Hal::SphericalAgent3D<Grid2>(pGrid),
          m_exampleProperty(0)
    {
    }

    int32_t ExampleProperty() const { return m_exampleProperty; }
    void    SetExampleProperty(int32_t exampleProperty) { m_exampleProperty = exampleProperty; }

private:
    int32_t m_exampleProperty;
};

class Grid2 : public Hal::AgentGrid3D<CellType2, Grid2>
{
public:
    Grid2(int32_t x, int32_t y, int32_t z)
        : Hal::AgentGrid3D<CellType2, Grid2>(x, y, z)
    {
    }
};

class TemplateMultiGrid
{
public:
    TemplateMultiGrid(int32_t x,
                      int32_t y,
                      int32_t z)
        : m_cells1(x, y, z),
          m_cells2(x, y, z),
          m_pde(x, y, z),
          m_hood(Hal::VonNeumannHood3D(false))
    {
        m_cells1.NewAgentSQ(x / 2, y / 2, z / 2);
        m_cells2.NewAgentSQ(x / 2, y / 2, z / 2);
    }

    void StepAll()
    {
        for (CellType1 &c1 : m_cells1)
        {
            c1.SetExampleProperty(Hal::Rgb(m_rng.Double(), m_rng.Double(), m_rng.Double()));
            int32_t n = c1.MapHood(m_hood.data(), static_cast<int32_t>(m_hood.size()));
            c1.MoveSQ(m_hood[m_rng.Int(n)]);
        }
        for (CellType2 &c2 : m_cells2)
        {
            c2.SetExampleProperty(Hal::Rgb(m_rng.Double(), m_rng.Double(), m_rng.Double()));
            c2.SetRadius(m_rng.Double());
            c2.SetVelocity(c2.XVel() + m_rng.Double(2) - 1, c2.YVel() + m_rng.Double(2) - 1, c2.ZVel() + m_rng.Double(2) - 1);
            c2.ForceMove();
            c2.ApplyFriction(0.0);
        }
        m_pde.Diffusion(0.1, [](int32_t x, int32_t, int32_t) { return (x < 0) ? 1.0 : 0.0; });
        m_pde.Update();
    }

    void DrawAll(Hal::Examples::Scene3D &scene)
    {
        scene.ClearBox(Hal::BLACK, Hal::WHITE);
        for (int32_t x = 0; x < m_pde.Xdim(); x++)
        {
            for (int32_t y = 0; y < m_pde.Ydim(); y++)
            {
                for (int32_t z = 0; z < m_pde.Zdim(); z++)
                {
                    scene.Sphere(x + 0.5, y + 0.5, z + 0.5, 0.1, Hal::HeatMapRgb(m_pde.Get(x, y, z)));
                }
            }
        }
        for (CellType1 &c1 : m_cells1)
        {
            scene.Sphere(c1.Xpt(), c1.Ypt(), c1.Zpt(), 0.5, c1.ExampleProperty());
        }
        for (CellType2 &c2 : m_cells2)
        {
            scene.Sphere(c2.Xpt(), c2.Ypt(), c2.Zpt(), c2.Radius(), c2.ExampleProperty());
        }
        scene.Render();
    }

    int32_t Dim() const { return m_cells1.Xdim(); }

private:
    Grid1                m_cells1;
    Grid2                m_cells2;
    Hal::PDEGrid3D       m_pde;
    Hal::Rand            m_rng;
    std::vector<int32_t> m_hood;
};

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        TemplateMultiGrid grids(11, 11, 11);
        Hal::Examples::GridWindow win("Multi-Grid Template", 500, 500, 1, !headless);
        Hal::Examples::Scene3D    scene(win, grids.Dim(), grids.Dim(), grids.Dim());

        int32_t steps = headless ? 50 : 1000000;
        for (int32_t i = 0; (i < steps) && !win.IsClosed(); i++)
        {
            scene.HandleInput();
            grids.StepAll();
            grids.DrawAll(scene);
            if (!headless)
            {
                win.TickPause(0);
            }
        }

        if (headless)
        {
            win.ToPng("templatemultigrid.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("TemplateMultiGrid failed: ") + e.what());
        return 1;
    }
}
