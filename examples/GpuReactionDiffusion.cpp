// Showcase: GPU-resident Gray-Scott reaction-diffusion. Two chemicals
// diffuse and feed each other on a 384x384 toroidal grid; their Turing
// instability grows the coral, maze, spot and mitosis patterns that make
// reaction-diffusion a staple of generative art. Every frame runs many
// simulation steps entirely on the GPU (GrayScottKernel2D::RunSteps) with
// no per-step host round-trip, which is exactly the shape that lets the
// GPU pay off (see docs/DESIGN.md's backend-abstraction section) — the
// whole time-stepped loop lives on the device, and the CPU only paints the
// result once per frame.
//
// The run is a tour: it cycles through several classic (feed, kill)
// regimes, reseeding and letting each settle, so one launch shows the
// range of patterns the same system produces.
//
// Usage:
//   GpuReactionDiffusion                live window, cycles all presets
//   GpuReactionDiffusion <preset>       live window, one preset
//                                       (coral | maze | mitosis | spots |
//                                        fingerprint | bubbles)
//   GpuReactionDiffusion headless [preset]
//                                       no window; writes gpu_reaction_
//                                       diffusion.mp4 and a final .png
//
// This is one of three GPU showcases (see also GpuInkFlow, GpuSwarm).

#include "Hal/Color.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/GrayScottKernel2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace GpuReactionDiffusionInternal
{
    // A named Gray-Scott regime: the feed/kill pair that selects a pattern
    // family, plus a display name for the CLI.
    struct Preset
    {
        const char *pName;
        float       feed;
        float       kill;
    };

    const Preset g_presets[] = {
        {"coral", 0.0545f, 0.0620f},
        {"maze", 0.0290f, 0.0570f},
        {"mitosis", 0.0367f, 0.0649f},
        {"spots", 0.0300f, 0.0620f},
        {"fingerprint", 0.0550f, 0.0620f},
        {"bubbles", 0.0980f, 0.0570f},
    };
    const int32_t g_presetCount = static_cast<int32_t>(sizeof(g_presets) / sizeof(g_presets[0]));

    // Smooth five-stop "ember" palette (deep indigo -> violet -> magenta ->
    // amber -> white), driven by the V concentration. A slow hue phase
    // rotates the anchors over time so a settled pattern keeps shimmering
    // rather than freezing. t is expected in [0, 1].
    int32_t EmberColor(double t,
                       double phase)
    {
        if (t < 0.0)
        {
            t = 0.0;
        }
        if (t > 1.0)
        {
            t = 1.0;
        }

        // Anchor colors as (r, g, b) in [0, 1].
        const double anchors[6][3] = {
            {0.02, 0.01, 0.09},
            {0.24, 0.03, 0.35},
            {0.72, 0.09, 0.42},
            {0.98, 0.55, 0.16},
            {1.00, 0.93, 0.62},
            {1.00, 1.00, 1.00},
        };
        const int32_t stops = 6;

        double scaled = t * (stops - 1);
        int32_t lo = static_cast<int32_t>(scaled);
        if (lo >= (stops - 1))
        {
            lo = stops - 2;
        }
        double f = scaled - lo;
        // Smoothstep for softer banding.
        double s = f * f * (3.0 - 2.0 * f);

        double r = anchors[lo][0] * (1.0 - s) + anchors[lo + 1][0] * s;
        double g = anchors[lo][1] * (1.0 - s) + anchors[lo + 1][1] * s;
        double b = anchors[lo][2] * (1.0 - s) + anchors[lo + 1][2] * s;

        // Gentle time-varying tint so the still image breathes.
        double warm = 0.5 + 0.5 * std::sin(phase);
        r = r * (0.88 + 0.12 * warm);
        b = b * (0.88 + 0.12 * (1.0 - warm));

        return Hal::Rgb(r, g, b);
    }

    // Seed a fresh pattern: U saturated, V zero, with a scatter of small
    // V-rich squares (U depleted) — the standard Gray-Scott perturbation.
    void SeedFields(std::vector<float> &u,
                    std::vector<float> &v,
                    int32_t             nx,
                    int32_t             ny,
                    Hal::Rand          &rand)
    {
        std::fill(u.begin(), u.end(), 1.0f);
        std::fill(v.begin(), v.end(), 0.0f);

        int32_t seeds = 24;
        for (int32_t s = 0; s < seeds; s++)
        {
            int32_t cx = rand.Int(nx);
            int32_t cy = rand.Int(ny);
            int32_t half = 3 + rand.Int(4);
            for (int32_t dx = -half; dx <= half; dx++)
            {
                for (int32_t dy = -half; dy <= half; dy++)
                {
                    int32_t x = ((cx + dx) % nx + nx) % nx;
                    int32_t y = ((cy + dy) % ny + ny) % ny;
                    size_t i = static_cast<size_t>(x * ny + y);
                    u[i] = 0.5f;
                    v[i] = 0.25f;
                }
            }
        }
    }
}

using namespace GpuReactionDiffusionInternal;

int main(int argc, char **argv)
{
    try
    {
        bool headless = false;
        int32_t fixedPreset = -1;
        for (int32_t a = 1; a < argc; a++)
        {
            if (std::strcmp(argv[a], "headless") == 0)
            {
                headless = true;
                continue;
            }
            for (int32_t p = 0; p < g_presetCount; p++)
            {
                if (std::strcmp(argv[a], g_presets[p].pName) == 0)
                {
                    fixedPreset = p;
                }
            }
        }

        const int32_t nx = 384;
        const int32_t ny = 384;
        const int32_t visScale = 2;
        const int32_t stepsPerFrame = 24;
        const float diffU = 1.0f;
        const float diffV = 0.5f;

        // Hold each preset for a fixed number of frames when cycling.
        const int32_t framesPerPreset = headless ? 60 : 900;
        const int32_t totalFrames = (fixedPreset >= 0) ? (headless ? 220 : 100000) : (framesPerPreset * g_presetCount);

        Hal::Gpu::ComputeContext context;
        Hal::Gpu::GrayScottKernel2D kernel(context, nx, ny);
        Hal::Rand rand;

        Hal::Examples::GridWindow win("GPU Reaction-Diffusion (Gray-Scott)", nx, ny, visScale, !headless);
        if (headless)
        {
            win.StartRecording("gpu_reaction_diffusion.mp4", 30);
        }

        std::vector<float> u(static_cast<size_t>(nx * ny));
        std::vector<float> v(static_cast<size_t>(nx * ny));

        int32_t activePreset = (fixedPreset >= 0) ? fixedPreset : 0;
        SeedFields(u, v, nx, ny, rand);

        for (int32_t frame = 0; (frame < totalFrames) && !win.IsClosed(); frame++)
        {
            // When cycling, advance to the next regime and reseed.
            if (fixedPreset < 0)
            {
                int32_t wanted = (frame / framesPerPreset) % g_presetCount;
                if (wanted != activePreset)
                {
                    activePreset = wanted;
                    SeedFields(u, v, nx, ny, rand);
                }
            }

            const Preset &preset = g_presets[activePreset];
            kernel.RunSteps(u, v, stepsPerFrame, preset.feed, preset.kill, diffU, diffV);

            double phase = frame * 0.02;
            for (int32_t x = 0; x < nx; x++)
            {
                for (int32_t y = 0; y < ny; y++)
                {
                    // V rarely exceeds ~0.4; stretch that into the palette.
                    double val = static_cast<double>(v[static_cast<size_t>(x * ny + y)]) * 2.5;
                    win.SetPix(x, y, EmberColor(val, phase));
                }
            }
            win.TickPause(headless ? 0 : 1);
        }

        if (headless)
        {
            win.StopRecording();
            win.ToPng("gpu_reaction_diffusion.png");
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_reaction_diffusion.mp4 and gpu_reaction_diffusion.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuReactionDiffusion failed: ") + e.what());
        return 1;
    }
}
