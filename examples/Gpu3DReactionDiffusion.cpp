// Showcase: a VOLUMETRIC Gray-Scott reaction-diffusion — the coral / sponge
// / gyroid morphologies of a Turing system grown in 3D and rendered as a
// cloud of depth-tested spheres on the hardware Vulkan renderer. Two
// chemicals diffuse and react across a 64^3 toroidal grid entirely on the
// GPU (GrayScott3DKernel, many resident steps per frame); each frame the
// iso-surface of the V field is extracted as a shell of spheres, tinted by
// height, and the whole structure slowly turns in space.
//
// This leans on the GPU twice over: a 3D compute kernel evolves a quarter-
// million voxels per step with no host round-trip inside a batch, and the
// hardware renderer depth-sorts the tens of thousands of shell spheres per
// pixel.
//
// It needs a live window (the Vulkan surface), but pass a frame count to
// record a clip via the renderer's opt-in frame capture (writes
// gpu_3d_reaction_diffusion.mp4 + a final .png).
//
// Usage:
//   Gpu3DReactionDiffusion                 live window, cycles presets
//   Gpu3DReactionDiffusion <preset>        coral | sponge | bubbles | maze
//   Gpu3DReactionDiffusion <preset> <n>    run n frames, recording mp4+png
//
// One of the GPU showcases (see also GpuReactionDiffusion, GpuFlowField3D,
// GpuAttractor3D, ...).

#include "Hal/Color.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/GrayScott3DKernel.h"
#include "Hal/Gpu/Scene3DRenderContext.h"
#include "Hal/Gpu/Window.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"
#include "Hal/Video/PngExport.h"

#include "common/Mp4Recorder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{
    struct Preset
    {
        const char *pName;
        float       feed;
        float       kill;
    };

    // feed/kill kept in the narrow 3D pattern-forming band (found by
    // sweeping): below it V floods the whole volume, above it V dies out.
    const Preset g_presets[] = {
        {"coral", 0.0610f, 0.0635f},
        {"sponge", 0.0580f, 0.0630f},
        {"bubbles", 0.0625f, 0.0640f},
        {"maze", 0.0555f, 0.0630f},
    };
    const int32_t g_presetCount = static_cast<int32_t>(sizeof(g_presets) / sizeof(g_presets[0]));

    // Iridescent height ramp: indigo -> teal -> green -> gold -> rose.
    int32_t HeightColor(double t)
    {
        if (t < 0.0)
        {
            t = 0.0;
        }
        if (t > 1.0)
        {
            t = 1.0;
        }
        const double anchors[5][3] = {
            {0.18, 0.05, 0.55},
            {0.05, 0.62, 0.75},
            {0.20, 0.85, 0.45},
            {0.98, 0.80, 0.20},
            {1.00, 0.35, 0.55},
        };
        const int32_t stops = 5;
        double scaled = t * (stops - 1);
        int32_t lo = static_cast<int32_t>(scaled);
        if (lo >= (stops - 1))
        {
            lo = stops - 2;
        }
        double f = scaled - lo;
        double s = f * f * (3.0 - 2.0 * f);
        double r = anchors[lo][0] * (1.0 - s) + anchors[lo + 1][0] * s;
        double g = anchors[lo][1] * (1.0 - s) + anchors[lo + 1][1] * s;
        double b = anchors[lo][2] * (1.0 - s) + anchors[lo + 1][2] * s;
        return Hal::Rgb(r, g, b);
    }

    int32_t Flatten(int32_t x,
                    int32_t y,
                    int32_t z,
                    int32_t ny,
                    int32_t nz)
    {
        return (x * ny + y) * nz + z;
    }

    void SeedFields(std::vector<float> &u,
                    std::vector<float> &v,
                    int32_t             n,
                    Hal::Rand          &rand)
    {
        std::fill(u.begin(), u.end(), 1.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        int32_t seeds = 30;
        for (int32_t s = 0; s < seeds; s++)
        {
            int32_t cx = n / 4 + rand.Int(n / 2);
            int32_t cy = n / 4 + rand.Int(n / 2);
            int32_t cz = n / 4 + rand.Int(n / 2);
            for (int32_t dx = -2; dx <= 2; dx++)
            {
                for (int32_t dy = -2; dy <= 2; dy++)
                {
                    for (int32_t dz = -2; dz <= 2; dz++)
                    {
                        int32_t x = ((cx + dx) % n + n) % n;
                        int32_t y = ((cy + dy) % n + n) % n;
                        int32_t z = ((cz + dz) % n + n) % n;
                        size_t i = static_cast<size_t>(Flatten(x, y, z, n, n));
                        u[i] = 0.5f;
                        v[i] = 0.25f;
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        int32_t fixedPreset = -1;
        int32_t maxFrames = 0;
        float feedOverride = -1.0f;
        float killOverride = -1.0f;
        for (int32_t a = 1; a < argc; a++)
        {
            if (std::strncmp(argv[a], "feed=", 5) == 0)
            {
                feedOverride = static_cast<float>(std::atof(argv[a] + 5));
                continue;
            }
            if (std::strncmp(argv[a], "kill=", 5) == 0)
            {
                killOverride = static_cast<float>(std::atof(argv[a] + 5));
                continue;
            }
            int32_t parsed = std::atoi(argv[a]);
            if (parsed > 0)
            {
                maxFrames = parsed;
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
        if ((feedOverride >= 0.0f) && (killOverride >= 0.0f))
        {
            fixedPreset = 0;
        }

        const int32_t n = 64;
        const int32_t stepsPerFrame = 18;
        const float diffU = 0.10f;
        const float diffV = 0.05f;
        const float threshold = 0.15f;
        const double box = 40.0;
        const double center = box / 2.0;
        const double voxel = box / n;

        const int32_t framesPerPreset = 260;
        const int32_t totalFrames = (fixedPreset >= 0) ? (maxFrames > 0 ? maxFrames : 100000) : (framesPerPreset * g_presetCount);

        Hal::Gpu::ComputeContext context;
        Hal::Gpu::GrayScott3DKernel kernel(context, n, n, n);
        Hal::Rand rand;

        std::vector<float> u(static_cast<size_t>(n * n * n));
        std::vector<float> v(static_cast<size_t>(n * n * n));
        int32_t activePreset = (fixedPreset >= 0) ? fixedPreset : 0;
        SeedFields(u, v, n, rand);

        Hal::Gpu::Window               window(960, 720, "GPU 3D Reaction-Diffusion");
        Hal::Gpu::Scene3DRenderContext scene(window, box, box, box);
        scene.SetCamera(0.6, 0.4, box * 1.7);

        bool capture = (maxFrames > 0);
        if (capture)
        {
            scene.SetCaptureEnabled(true);
        }
        std::unique_ptr<Hal::Examples::Mp4Recorder> recorder;
        std::vector<uint8_t> lastFrame;
        int32_t lastWidth = 0;
        int32_t lastHeight = 0;

        double yaw = 0.6;
        int32_t frame = 0;
        while (!window.ShouldClose() && (frame < totalFrames))
        {
            window.PollEvents();

            if (fixedPreset < 0)
            {
                int32_t wanted = (frame / framesPerPreset) % g_presetCount;
                if (wanted != activePreset)
                {
                    activePreset = wanted;
                    SeedFields(u, v, n, rand);
                }
            }

            const Preset &preset = g_presets[activePreset];
            float feed = (feedOverride >= 0.0f) ? feedOverride : preset.feed;
            float kill = (killOverride >= 0.0f) ? killOverride : preset.kill;
            kernel.RunSteps(u, v, stepsPerFrame, feed, kill, diffU, diffV);

            yaw += 0.004;
            scene.SetCamera(yaw, 0.4 + 0.1 * std::sin(frame * 0.006), box * 1.7);
            scene.Clear(Hal::Rgb(0.02, 0.02, 0.05));

            // Extract the V iso-surface as a shell of spheres: a filled
            // voxel with at least one empty face neighbour is on the
            // surface. (Interior voxels are hidden anyway, so skipping them
            // is both faster and cleaner.)
            for (int32_t x = 0; x < n; x++)
            {
                for (int32_t y = 0; y < n; y++)
                {
                    for (int32_t z = 0; z < n; z++)
                    {
                        float value = v[static_cast<size_t>(Flatten(x, y, z, n, n))];
                        if (value < threshold)
                        {
                            continue;
                        }
                        int32_t xm = (x - 1 + n) % n;
                        int32_t xp = (x + 1) % n;
                        int32_t ym = (y - 1 + n) % n;
                        int32_t yp = (y + 1) % n;
                        int32_t zm = (z - 1 + n) % n;
                        int32_t zp = (z + 1) % n;
                        bool surface =
                            (v[static_cast<size_t>(Flatten(xm, y, z, n, n))] < threshold) ||
                            (v[static_cast<size_t>(Flatten(xp, y, z, n, n))] < threshold) ||
                            (v[static_cast<size_t>(Flatten(x, ym, z, n, n))] < threshold) ||
                            (v[static_cast<size_t>(Flatten(x, yp, z, n, n))] < threshold) ||
                            (v[static_cast<size_t>(Flatten(x, y, zm, n, n))] < threshold) ||
                            (v[static_cast<size_t>(Flatten(x, y, zp, n, n))] < threshold);
                        if (!surface)
                        {
                            continue;
                        }
                        double wx = (x + 0.5) * voxel;
                        double wy = (y + 0.5) * voxel;
                        double wz = (z + 0.5) * voxel;
                        scene.Sphere(wx, wy, wz, voxel * 0.62, HeightColor(static_cast<double>(z) / (n - 1)));
                    }
                }
            }

            bool presented = scene.Render();
            if (capture && presented)
            {
                if (scene.ReadCapturedFrame(lastFrame, lastWidth, lastHeight))
                {
                    if (!recorder)
                    {
                        recorder = std::make_unique<Hal::Examples::Mp4Recorder>("gpu_3d_reaction_diffusion.mp4", lastWidth, lastHeight, 30, 1);
                    }
                    recorder->AddFrame(lastFrame);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
            frame++;
        }

        if (capture && !lastFrame.empty())
        {
            recorder.reset();
            Hal::ExportPng("gpu_3d_reaction_diffusion.png", lastWidth, lastHeight, lastFrame);
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_3d_reaction_diffusion.mp4 and gpu_3d_reaction_diffusion.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("Gpu3DReactionDiffusion failed: ") + e.what());
        return 1;
    }
}
