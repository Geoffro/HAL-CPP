// Two-channel ("N-player") Lenia: each channel's growth depends on the
// convolved potentials of both channels through a payoff matrix. Panels
// show each channel's state (greyscale) and all four pairwise convolved
// fields K_ij * A_j (jet).
//
// Live window by default; pass "headless" to run fewer steps and dump the
// final panels to orbiumnplayer.png.

#include "Hal/Color.h"
#include "Hal/Grids/Grid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"
#include "Hal/Tools/LeniaNPlayer.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static const int32_t R = 13;
static const double  ALPHA = 4.0;

// Ring interaction kernel (between radii 2.5 and 7.5).
static double Kc2(int32_t, int32_t, double r)
{
    return (r > 7.5) ? 0.0 : (r < 2.5) ? 0.0 : 1.0;
}

// Payoff-coupled density-dependent growth for two channels.
static const double growthGamma[2] = {1.0, 1.0};
static const double payoff[2][2] = {{1.0, 0.5}, {0.0, 1.0}};

static double G2(int32_t type,
                 const std::vector<double> &x)
{
    double sum = 0.0;
    for (size_t j = 0; j < x.size(); j++)
    {
        sum += payoff[type][j] * x[j];
    }
    return growthGamma[type] * x[type] * (1.0 - sum);
}

static void RandomIC(Hal::LeniaNPlayer &model,
                     Hal::Rand         &rng)
{
    for (int32_t i = 0; i < model.Length(); i++)
    {
        model.Set(0, i, rng.Double() / 10);
        model.Set(1, i, rng.Double() / 10);
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);
        (void)R;
        (void)ALPHA;

        const int32_t sideLenExp = 6;
        const int32_t sideLen = 1 << sideLenExp; // 64
        const int32_t nPanels = 6;               // 2 fields + 4 convolved
        const int32_t panel = sideLen + 1;

        Hal::LeniaNPlayer model(sideLenExp, 2, 0.2, Kc2, G2);
        Hal::Rand         rng;
        RandomIC(model, rng);

        Hal::Examples::GridWindow win("Lenia N-Player", panel * nPanels, sideLen, headless ? 3 : 4, !headless);

        int32_t totalTime = headless ? 200 : 10000;
        while ((model.GetTick() < totalTime) && !win.IsClosed())
        {
            // Channel state fields (panels 0, 1).
            for (int32_t p = 0; p < 2; p++)
            {
                const Hal::Grid2D<double> &field = model.GetField(p);
                for (int32_t x = 0; x < sideLen; x++)
                {
                    for (int32_t y = 0; y < sideLen; y++)
                    {
                        win.SetPix(p * panel + x, y, Hal::GreyScale(field.Get(x, y)));
                    }
                }
            }
            // Pairwise convolved fields (panels 2..5).
            for (int32_t i = 0; i < 2; i++)
            {
                for (int32_t j = 0; j < 2; j++)
                {
                    int32_t panelIdx = 2 + i * 2 + j;
                    for (int32_t x = 0; x < sideLen; x++)
                    {
                        for (int32_t y = 0; y < sideLen; y++)
                        {
                            double v = model.GetConvolvedFieldVal(i, j, x * sideLen + y);
                            win.SetPix(panelIdx * panel + x, y, Hal::HeatMapJet(v, -1.0, 1.0));
                        }
                    }
                }
            }
            win.TickPause(headless ? 0 : 16);
            model.Update();
        }

        if (headless)
        {
            win.ToPng("orbiumnplayer.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("OrbiumNPlayer failed: ") + e.what());
        return 1;
    }
}
