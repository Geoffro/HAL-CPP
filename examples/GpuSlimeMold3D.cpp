// Showcase: a VOLUMETRIC Physarum ("slime mold") — the 3D hybrid. This is
// the thing HAL is actually for, pushed onto the GPU: interacting agents
// coupled to a diffusible chemical field, in 3D, rendered on the hardware
// Vulkan renderer. 200,000 agents drift through an empty 96^3 volume, each
// sensing the chemical field in a forward cone, steering up its gradient,
// stepping, and depositing a little chemical of its own. That field is
// blurred and faded on the GPU every step, so nearby agents reinforce each
// other's trails and unused ones evaporate — and out of those local rules a
// living 3D transport network grows itself, endlessly restructuring in space.
//
// Everything but the drawing lives on the GPU and stays there: the agent
// sense/steer/move/deposit pass AND the field diffuse/decay pass are compute
// kernels (SlimeMold3DKernel), and the chemical field never leaves the device
// between frames. Only the agent positions round-trip to the host, to be
// drawn as tens of thousands of depth-tested spheres (Scene3DRenderContext),
// tinted by the concentration each agent sits in. So it leans on the GPU on
// both sides at once: a big hybrid compute workload feeding a big instanced
// render workload.
//
// It needs a live window (the Vulkan surface), but pass a frame count to
// record a clip via the renderer's opt-in frame capture (writes
// gpu_slime_mold_3d.mp4 + a final .png).
//
// Usage:
//   GpuSlimeMold3D                    live window, palette cycles
//   GpuSlimeMold3D <palette>          gold | ember | spore | ice
//   GpuSlimeMold3D <palette> <frames> run <frames> frames, recording mp4+png
//   GpuSlimeMold3D <palette> <frames> <agents>
//                                     override the agent count too
//
// One of the GPU showcases (see also GpuSlimeMold, GpuFlowField3D,
// Gpu3DReactionDiffusion, GpuAttractor3D, ...).

#include "Hal/Color.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/Scene3DRenderContext.h"
#include "Hal/Gpu/SlimeMold3DKernel.h"
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

namespace GpuSlimeMold3DInternal
{
    // A named color ramp for the chemical concentration (5 anchor stops, near
    // black first so the empty void stays dark). t in [0, 1].
    struct Palette
    {
        const char *pName;
        double      anchors[5][3];
    };

    const Palette g_palettes[] = {
        {"gold", {{0.01, 0.01, 0.02}, {0.20, 0.06, 0.00}, {0.75, 0.35, 0.03}, {1.00, 0.78, 0.25}, {1.00, 1.00, 0.90}}},
        {"ember", {{0.02, 0.00, 0.04}, {0.30, 0.02, 0.20}, {0.80, 0.10, 0.24}, {1.00, 0.52, 0.16}, {1.00, 0.96, 0.80}}},
        {"spore", {{0.00, 0.02, 0.03}, {0.02, 0.18, 0.20}, {0.05, 0.55, 0.42}, {0.55, 0.95, 0.45}, {0.92, 1.00, 0.85}}},
        {"ice", {{0.00, 0.02, 0.05}, {0.03, 0.10, 0.35}, {0.10, 0.45, 0.80}, {0.45, 0.85, 1.00}, {0.95, 1.00, 1.00}}},
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
}

using namespace GpuSlimeMold3DInternal;

int main(int argc, char **argv)
{
    try
    {
        int32_t fixedPalette = -1;
        int32_t maxFrames = 0;
        int32_t requestedCount = 200000;
        // Numeric args, in order: first is the frame count (triggers capture),
        // a second overrides the agent count.
        std::vector<int32_t> numbers;
        for (int32_t a = 1; a < argc; a++)
        {
            bool matched = false;
            for (int32_t p = 0; p < g_paletteCount; p++)
            {
                if (std::strcmp(argv[a], g_palettes[p].pName) == 0)
                {
                    fixedPalette = p;
                    matched = true;
                }
            }
            if (matched)
            {
                continue;
            }
            int32_t parsed = std::atoi(argv[a]);
            if (parsed > 0)
            {
                numbers.push_back(parsed);
            }
        }
        if (numbers.size() == 1)
        {
            maxFrames = numbers[0];
        }
        else if (numbers.size() >= 2)
        {
            maxFrames = numbers[0];
            requestedCount = numbers[1];
        }

        const int32_t n = 96;
        const int32_t agentCount = requestedCount;
        const int32_t stepsPerFrame = 2;
        const double  box = 40.0;
        const double  voxel = box / n;

        // Physarum rules in grid units. diffuse stays under the 3D explicit
        // stability limit (1/6); decay < 1 so unused trails evaporate.
        Hal::Gpu::SlimeMold3DParams params = {};
        params.stepSize = 0.9f;
        params.sensorAngle = 0.55f;
        params.sensorDist = 5.0f;
        params.turnRate = 0.32f;
        params.depositAmount = 5.0f;
        params.jitter = 0.12f;
        params.diffuse = 0.13f;
        params.decay = 0.90f;
        params.seed = 1u;

        const int32_t framesPerPalette = 320;
        const int32_t totalFrames = (fixedPalette >= 0) ? (maxFrames > 0 ? maxFrames : 100000) : (framesPerPalette * g_paletteCount);

        Hal::Gpu::ComputeContext    context;
        Hal::Gpu::SlimeMold3DKernel kernel(context, n, n, n, agentCount);
        Hal::Rand                   rand;

        // Seed agents uniformly through the volume with random headings, so
        // the network self-organizes throughout the whole box.
        std::vector<float> agents(static_cast<size_t>(agentCount) * 8);
        for (int32_t i = 0; i < agentCount; i++)
        {
            double dir[3] = {};
            rand.RandomPointOnSphereEdge(1.0, dir);
            agents[static_cast<size_t>(8 * i)] = static_cast<float>(rand.Double(n));
            agents[static_cast<size_t>(8 * i + 1)] = static_cast<float>(rand.Double(n));
            agents[static_cast<size_t>(8 * i + 2)] = static_cast<float>(rand.Double(n));
            agents[static_cast<size_t>(8 * i + 3)] = 0.0f;
            agents[static_cast<size_t>(8 * i + 4)] = static_cast<float>(dir[0]);
            agents[static_cast<size_t>(8 * i + 5)] = static_cast<float>(dir[1]);
            agents[static_cast<size_t>(8 * i + 6)] = static_cast<float>(dir[2]);
            agents[static_cast<size_t>(8 * i + 7)] = 0.0f;
        }

        Hal::Gpu::Window               window(960, 720, "GPU Slime Mold 3D (volumetric Physarum)");
        Hal::Gpu::Scene3DRenderContext scene(window, box, box, box);
        scene.SetCamera(0.6, 0.35, box * 1.7);

        bool capture = (maxFrames > 0);
        if (capture)
        {
            scene.SetCaptureEnabled(true);
        }
        std::unique_ptr<Hal::Examples::Mp4Recorder> recorder;
        std::vector<uint8_t>                         lastFrame;
        int32_t                                      lastWidth = 0;
        int32_t                                      lastHeight = 0;

        int32_t activePalette = (fixedPalette >= 0) ? fixedPalette : 0;
        double  sphereRadius = voxel * 0.5;
        double  yaw = 0.6;
        int32_t frame = 0;
        while (!window.ShouldClose() && (frame < totalFrames))
        {
            window.PollEvents();

            if (fixedPalette < 0)
            {
                activePalette = (frame / framesPerPalette) % g_paletteCount;
            }

            params.seed = static_cast<uint32_t>(frame * stepsPerFrame + 1);
            kernel.RunSteps(agents, agentCount, stepsPerFrame, params);

            yaw += 0.0035;
            scene.SetCamera(yaw, 0.35 + 0.12 * std::sin(frame * 0.004), box * 1.7);
            scene.Clear(Hal::Rgb(0.01, 0.01, 0.03));

            const Palette &palette = g_palettes[activePalette];
            for (int32_t i = 0; i < agentCount; i++)
            {
                double px = agents[static_cast<size_t>(8 * i)] * voxel;
                double py = agents[static_cast<size_t>(8 * i + 1)] * voxel;
                double pz = agents[static_cast<size_t>(8 * i + 2)] * voxel;
                double conc = agents[static_cast<size_t>(8 * i + 3)];
                double t = 1.0 - std::exp(-conc * 0.06);
                scene.Sphere(px, py, pz, sphereRadius, RampColor(palette, t));
            }

            bool presented = scene.Render();
            if (capture && presented)
            {
                if (scene.ReadCapturedFrame(lastFrame, lastWidth, lastHeight))
                {
                    if (!recorder)
                    {
                        recorder = std::make_unique<Hal::Examples::Mp4Recorder>("gpu_slime_mold_3d.mp4", lastWidth, lastHeight, 30, 1);
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
            Hal::ExportPng("gpu_slime_mold_3d.png", lastWidth, lastHeight, lastFrame);
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_slime_mold_3d.mp4 and gpu_slime_mold_3d.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuSlimeMold3D failed: ") + e.what());
        return 1;
    }
}
