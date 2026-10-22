// A 3D diffusible released at the center of a periodic box, transported
// by advection along x and z while spreading by diffusion. The mid-z slice
// is drawn as a heat map.
//
// Live window by default; pass "headless" to run fewer steps and dump the
// final slice to diffusionadvection3d.png.

#include "Hal/Color.h"
#include "Hal/Grids/PDEGrid3D.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t dim = 20;
        Hal::Examples::GridWindow win("3D Advection (XY slice)", dim, dim, 10, !headless);
        Hal::PDEGrid3D grid(dim, dim, dim, true, true, true);
        grid.Set(dim / 2, dim / 2, dim / 2, 1.0);

        int32_t zSlice = dim / 2;
        int32_t steps = headless ? 200 : 1000000;
        for (int32_t step = 0; (step < steps) && !win.IsClosed(); step++)
        {
            grid.Advection(0.1, 0.0, 0.1, 0.0);
            grid.Diffusion(0.13);
            grid.Update();
            for (int32_t x = 0; x < dim; x++)
            {
                for (int32_t y = 0; y < dim; y++)
                {
                    win.SetPix(x, y, Hal::HeatMapBgr(grid.Get(x, y, zSlice) * 1000));
                }
            }
            win.TickPause(headless ? 0 : 100);
        }

        if (headless)
        {
            win.ToPng("diffusionadvection3d.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("DiffusionAdvection3D failed: ") + e.what());
        return 1;
    }
}
