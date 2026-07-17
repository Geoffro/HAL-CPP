// Showcase: a Physarum polycephalum ("slime mold") transport network —
// the emergent, endlessly restructuring web of veins that a real slime
// mold grows to connect food sources, and a favorite of generative
// artists. 220,000 agents each sense the chemical trail just ahead of
// them (three sensors: left, center, right), steer toward wherever it is
// strongest, step forward, and deposit a little trail of their own. That
// deposited field is then blurred and faded on the GPU
// (DiffusionKernel2D::RunSteps) — the diffusion that lets nearby agents
// reinforce each other's paths and unused ones evaporate. Out of those
// three trivial rules per agent, a living network organizes itself.
//
// The agent sense/steer/move/deposit pass is the cheap-per-agent,
// massively-parallel part (a few trail samples each); the trail diffusion
// is the field operation, done on the GPU.
//
// Usage:
//   GpuSlimeMold                live window (palette cycles)
//   GpuSlimeMold <palette>      live window, one palette
//                               (gold | ember | spore | ice)
//   GpuSlimeMold headless [palette]
//                               no window; writes gpu_slime_mold.mp4 + png
//   GpuSlimeMold <N> ...        override the agent count
//
// One of the GPU showcases (see also GpuReactionDiffusion, GpuInkFlow,
// GpuSwarm, GpuWaveField, GpuAttractor3D).

#include "Hal/Color.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/DiffusionKernel2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace GpuSlimeMoldInternal
{
    // One slime agent: a position and a heading. Trail is deposited at the
    // position; the three sensors are placed relative to the heading.
    struct Agent
    {
        double x;
        double y;
        double heading;
    };

    // A named color ramp for the trail intensity (5 anchor stops, black
    // first so empty space stays dark). t in [0, 1].
    struct Palette
    {
        const char  *pName;
        double        anchors[5][3];
    };

    const Palette g_palettes[] = {
        {"gold", {{0.00, 0.00, 0.00}, {0.20, 0.06, 0.00}, {0.75, 0.35, 0.03}, {1.00, 0.78, 0.25}, {1.00, 1.00, 0.90}}},
        {"ember", {{0.01, 0.00, 0.03}, {0.30, 0.02, 0.20}, {0.80, 0.10, 0.24}, {1.00, 0.52, 0.16}, {1.00, 0.96, 0.80}}},
        {"spore", {{0.00, 0.01, 0.02}, {0.02, 0.18, 0.20}, {0.05, 0.55, 0.42}, {0.55, 0.95, 0.45}, {0.92, 1.00, 0.85}}},
        {"ice", {{0.00, 0.01, 0.04}, {0.03, 0.10, 0.35}, {0.10, 0.45, 0.80}, {0.45, 0.85, 1.00}, {0.95, 1.00, 1.00}}},
    };
    const int32_t g_paletteCount = static_cast<int32_t>(sizeof(g_palettes) / sizeof(g_palettes[0]));

    int32_t RampColor(const Palette &palette,
                      double          t)
    {
        if (t < 0.0)
        {
            t = 0.0;
        }
        if (t > 1.0)
        {
            t = 1.0;
        }
        const int32_t stops = 5;
        double scaled = t * (stops - 1);
        int32_t lo = static_cast<int32_t>(scaled);
        if (lo >= (stops - 1))
        {
            lo = stops - 2;
        }
        double f = scaled - lo;
        double s = f * f * (3.0 - 2.0 * f);
        double r = palette.anchors[lo][0] * (1.0 - s) + palette.anchors[lo + 1][0] * s;
        double g = palette.anchors[lo][1] * (1.0 - s) + palette.anchors[lo + 1][1] * s;
        double b = palette.anchors[lo][2] * (1.0 - s) + palette.anchors[lo + 1][2] * s;
        return Hal::Rgb(r, g, b);
    }

    // Nearest-cell trail sample with toroidal wrap.
    float SampleTrail(const std::vector<float> &trail,
                      int32_t                   nx,
                      int32_t                   ny,
                      double                    x,
                      double                    y)
    {
        int32_t ix = ((static_cast<int32_t>(std::lround(x)) % nx) + nx) % nx;
        int32_t iy = ((static_cast<int32_t>(std::lround(y)) % ny) + ny) % ny;
        return trail[static_cast<size_t>(ix * ny + iy)];
    }
}

using namespace GpuSlimeMoldInternal;

int main(int argc, char **argv)
{
    try
    {
        bool headless = false;
        int32_t fixedPalette = -1;
        int32_t requestedCount = 340000;
        for (int32_t a = 1; a < argc; a++)
        {
            if (std::strcmp(argv[a], "headless") == 0)
            {
                headless = true;
                continue;
            }
            bool matched = false;
            for (int32_t p = 0; p < g_paletteCount; p++)
            {
                if (std::strcmp(argv[a], g_palettes[p].pName) == 0)
                {
                    fixedPalette = p;
                    matched = true;
                }
            }
            if (!matched)
            {
                int32_t parsed = std::atoi(argv[a]);
                if (parsed > 0)
                {
                    requestedCount = parsed;
                }
            }
        }

        const int32_t nx = 512;
        const int32_t ny = 512;
        const int32_t visScale = 1;
        const int32_t numAgents = requestedCount;

        // Physarum rules (Jeff Jones' parameters, lightly tuned).
        const double sensorAngle = 0.40;
        const double sensorDistance = 6.0;
        const double turnAngle = 0.42;
        const double stepSize = 1.0;
        const float depositAmount = 4.0f;
        const float diffRate = 0.16f;
        const double decay = 0.93;
        const int32_t framesPerPalette = headless ? 90 : 700;
        const int32_t totalFrames = (fixedPalette >= 0) ? (headless ? 320 : 100000) : (framesPerPalette * g_paletteCount);

        Hal::Gpu::ComputeContext context;
        Hal::Gpu::DiffusionKernel2D kernel(context, nx, ny);
        Hal::Rand rand;

        std::vector<float> trail(static_cast<size_t>(nx * ny), 0.0f);
        std::vector<Agent> agents(static_cast<size_t>(numAgents));

        // Seed agents uniformly over the whole field with random headings,
        // so the network self-organizes edge to edge and fills the frame.
        for (int32_t i = 0; i < numAgents; i++)
        {
            agents[static_cast<size_t>(i)].x = rand.Double(nx);
            agents[static_cast<size_t>(i)].y = rand.Double(ny);
            agents[static_cast<size_t>(i)].heading = rand.Double(2.0 * M_PI);
        }

        Hal::Examples::GridWindow win("GPU Slime Mold (Physarum network)", nx, ny, visScale, !headless);
        if (headless)
        {
            win.StartRecording("gpu_slime_mold.mp4", 30);
        }

        int32_t activePalette = (fixedPalette >= 0) ? fixedPalette : 0;

        for (int32_t frame = 0; (frame < totalFrames) && !win.IsClosed(); frame++)
        {
            if (fixedPalette < 0)
            {
                activePalette = (frame / framesPerPalette) % g_paletteCount;
            }

            // Sense, steer, move, deposit — one pass over all agents.
            for (int32_t i = 0; i < numAgents; i++)
            {
                Agent &agent = agents[static_cast<size_t>(i)];

                double fx = agent.x + std::cos(agent.heading) * sensorDistance;
                double fy = agent.y + std::sin(agent.heading) * sensorDistance;
                double lx = agent.x + std::cos(agent.heading + sensorAngle) * sensorDistance;
                double ly = agent.y + std::sin(agent.heading + sensorAngle) * sensorDistance;
                double rx = agent.x + std::cos(agent.heading - sensorAngle) * sensorDistance;
                double ry = agent.y + std::sin(agent.heading - sensorAngle) * sensorDistance;

                float senseF = SampleTrail(trail, nx, ny, fx, fy);
                float senseL = SampleTrail(trail, nx, ny, lx, ly);
                float senseR = SampleTrail(trail, nx, ny, rx, ry);

                if ((senseF >= senseL) && (senseF >= senseR))
                {
                    // Straight ahead is strongest: keep heading.
                }
                else if ((senseF < senseL) && (senseF < senseR))
                {
                    // Both sides beat center: turn randomly one way.
                    agent.heading += rand.Bool() ? turnAngle : -turnAngle;
                }
                else if (senseL < senseR)
                {
                    agent.heading -= turnAngle;
                }
                else
                {
                    agent.heading += turnAngle;
                }

                agent.x += std::cos(agent.heading) * stepSize;
                agent.y += std::sin(agent.heading) * stepSize;
                agent.x = std::fmod(agent.x + nx, static_cast<double>(nx));
                agent.y = std::fmod(agent.y + ny, static_cast<double>(ny));

                int32_t dxCell = static_cast<int32_t>(agent.x);
                int32_t dyCell = static_cast<int32_t>(agent.y);
                trail[static_cast<size_t>(dxCell * ny + dyCell)] += depositAmount;
            }

            // Blur the trail on the GPU, then evaporate.
            kernel.RunSteps(trail, 1, diffRate, true, true);

            const Palette &palette = g_palettes[activePalette];
            for (int32_t x = 0; x < nx; x++)
            {
                for (int32_t y = 0; y < ny; y++)
                {
                    size_t i = static_cast<size_t>(x * ny + y);
                    trail[i] = static_cast<float>(trail[i] * decay);
                    double t = 1.0 - std::exp(-static_cast<double>(trail[i]) * 0.22);
                    win.SetPix(x, y, RampColor(palette, t));
                }
            }
            win.TickPause(headless ? 0 : 1);
        }

        if (headless)
        {
            win.StopRecording();
            win.ToPng("gpu_slime_mold.png");
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_slime_mold.mp4 and gpu_slime_mold.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuSlimeMold failed: ") + e.what());
        return 1;
    }
}
