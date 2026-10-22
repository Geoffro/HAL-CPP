// The Lenia "orbium" glider: a continuous cellular automaton driven by an
// FFT convolution (Lenia1Player). A smooth ring kernel K convolves the
// state A into a potential U, and a Gaussian growth map G(U) updates A.
// Four panels sit side by side: the state A, the convolved field K*A, the
// growth field G(K*A), and the kernel K.
//
// Live window by default; pass "headless" to run a bounded number of steps
// and dump the final frame to orbium.png.

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"
#include "Hal/Tools/Lenia1Player.h"

#include "common/GridWindow.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

namespace OrbiumExample
{
const int32_t R = 13;
const double  ALPHA = 4.0;
const double  MU = 0.15;
const double  SIGMA = 0.017;

double Kc(double r)
{
    // Exponential ring kernel (from the Lenia paper).
    return (r > R) ? 0.0 : std::exp(ALPHA - ALPHA / (4.0 * r / R * (1.0 - r / R)));
}

double G(double density)
{
    // Gaussian growth map.
    double l = std::abs(density - MU);
    double k = 2.0 * SIGMA * SIGMA;
    return 2.0 * std::exp(-(l * l) / k) - 1.0;
}

// 20x20 orbium initial condition (row-major).
const double IC[20][20] = {
        {0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0},
        {0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0},
        {0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0},
        {0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0},
        {0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0},
        {0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0},
        {0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0},
        {0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0},
        {0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0},
        {0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07},
        {0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11},
        {0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1},
        {0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05},
        {0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01},
        {0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0},
        {0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0},
        {0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0},
        {0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0},
        {0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0},
        {0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0}};

void AddOrbium(Hal::Lenia1Player &model)
{
    for (int32_t x = 0; x < 20; x++)
    {
        for (int32_t y = 0; y < 20; y++)
        {
            model.Set(x + model.Xdim() / 2 - 10, y + model.Ydim() / 2 - 10, IC[x][y]);
        }
    }
}
}
using namespace OrbiumExample;

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t sideLenExp = 6;
        const int32_t sideLen = 1 << sideLenExp; // 64
        const int32_t sf = 5;

        Hal::Lenia1Player model(sideLenExp, 0.2, Kc, G);

        // Four panels: A | K*A | G(U) | K, each sideLen wide + 1 separator.
        const int32_t panel = sideLen + 1;
        Hal::Examples::GridWindow win("Lenia (Orbium)", panel * 4, sideLen, sf, !headless);

        // Kernel panel is static; draw it once.
        for (int32_t x = 0; x < sideLen; x++)
        {
            for (int32_t y = 0; y < sideLen; y++)
            {
                win.SetPix(panel * 3 + x, y, Hal::GreyScale(model.GetKernelVal(x, y)));
            }
        }

        AddOrbium(model);

        int32_t totalTime = headless ? 200 : 10000;
        while ((model.GetTick() < totalTime) && !win.IsClosed())
        {
            model.Update();
            for (int32_t x = 0; x < sideLen; x++)
            {
                for (int32_t y = 0; y < sideLen; y++)
                {
                    double conv = model.GetConvolvedFieldVal(x, y);
                    win.SetPix(x, y, Hal::GreyScale(model.Get(x, y)));
                    win.SetPix(panel + x, y, Hal::HeatMapJet(conv, -1.0, 1.0));
                    win.SetPix(panel * 2 + x, y, Hal::HeatMapJet(G(conv), -1.0, 1.0));
                }
            }
            win.TickPause(headless ? 0 : 16);
        }

        if (headless)
        {
            win.ToPng("orbium.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("Orbium failed: ") + e.what());
        return 1;
    }
}
