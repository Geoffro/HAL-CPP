// Showcase: a luminous 3D nebula of 120,000 tracer particles woven through a
// chaotic Arnold-Beltrami-Childress (ABC) flow. The flow is steady and
// divergence-free, but its streamlines mix chaotically, so the tracers braid
// into glowing tubes and sheets that fill the volume. Every particle is
// advected on the GPU (FlowFieldKernel3D, a few resident RK2 sub-steps per
// frame) and drawn as a depth-tested sphere on the hardware Vulkan renderer,
// tinted by its speed; the whole cloud slowly turns in space.
//
// This is the particle-system counterpart to the volumetric reaction-
// diffusion showcase: a big GPU compute workload (120k independent
// advections per sub-step) feeding a big GPU render workload (120k
// depth-sorted instances).
//
// It needs a live window (the Vulkan surface), but pass a frame count to
// record a clip via the renderer's opt-in frame capture (writes
// gpu_flow_field_3d.mp4 + a final .png).
//
// Usage:
//   GpuFlowField3D                live window
//   GpuFlowField3D <frames>       run <frames> frames, recording mp4+png
//   GpuFlowField3D <count> <frames>
//                                 override the particle count too
//
// One of the GPU showcases (see also Gpu3DReactionDiffusion, GpuAttractor3D,
// GpuSwarm, ...).

#include "Hal/Color.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/FlowFieldKernel3D.h"
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
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace GpuFlowField3DInternal
{
    // Electric speed palette: deep blue -> cyan -> white -> warm gold.
    int32_t SpeedColor(double t)
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
            {0.03, 0.05, 0.30},
            {0.10, 0.55, 0.95},
            {0.75, 0.98, 1.00},
            {1.00, 0.90, 0.55},
            {1.00, 0.70, 0.25},
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
}

using namespace GpuFlowField3DInternal;

int main(int argc, char **argv)
{
    try
    {
        int32_t particleCount = 120000;
        int32_t maxFrames = 0;
        // First positive int is the frame count; a second (larger) one is
        // taken as an explicit particle count override.
        std::vector<int32_t> numbers;
        for (int32_t a = 1; a < argc; a++)
        {
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
            particleCount = numbers[0];
            maxFrames = numbers[1];
        }

        const double twoPi = 6.28318530717958647692;
        const double box = 40.0;
        const double scale = box / twoPi;
        const float coeffA = 1.7f;
        const float coeffB = 1.3f;
        const float coeffC = 1.0f;
        const float dt = 0.018f;
        const int32_t substeps = 2;
        const double maxSpeed = static_cast<double>(coeffA) + static_cast<double>(coeffB) + static_cast<double>(coeffC);

        Hal::Gpu::ComputeContext context;
        Hal::Gpu::FlowFieldKernel3D kernel(context, particleCount);
        Hal::Rand rand;

        // Seed inside a centered sphere (not the whole periodic box) so the
        // cloud starts as a rounded blob and the ABC flow braids it into
        // tubes and sheets, instead of a uniformly filled cube.
        const double half = twoPi * 0.5;
        const double seedRadius = twoPi * 0.32;
        std::vector<float> positions(static_cast<size_t>(particleCount) * 4);
        for (int32_t i = 0; i < particleCount; i++)
        {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            for (;;)
            {
                x = rand.Double(twoPi) - half;
                y = rand.Double(twoPi) - half;
                z = rand.Double(twoPi) - half;
                if ((x * x + y * y + z * z) <= (seedRadius * seedRadius))
                {
                    break;
                }
            }
            positions[static_cast<size_t>(4 * i)] = static_cast<float>(x + half);
            positions[static_cast<size_t>(4 * i + 1)] = static_cast<float>(y + half);
            positions[static_cast<size_t>(4 * i + 2)] = static_cast<float>(z + half);
            positions[static_cast<size_t>(4 * i + 3)] = 0.0f;
        }

        Hal::Gpu::Window               window(960, 720, "GPU Flow Field 3D (ABC nebula)");
        Hal::Gpu::Scene3DRenderContext scene(window, box, box, box);
        scene.SetCamera(0.6, 0.3, box * 1.6);

        bool capture = (maxFrames > 0);
        if (capture)
        {
            scene.SetCaptureEnabled(true);
        }
        std::unique_ptr<Hal::Examples::Mp4Recorder> recorder;
        std::vector<uint8_t> lastFrame;
        int32_t lastWidth = 0;
        int32_t lastHeight = 0;

        double sphereRadius = box * 0.0016;
        double yaw = 0.6;
        int32_t frame = 0;
        while (!window.ShouldClose() && (!capture || (frame < maxFrames)))
        {
            window.PollEvents();

            kernel.RunSteps(positions, substeps, coeffA, coeffB, coeffC, dt);

            yaw += 0.005;
            scene.SetCamera(yaw, 0.3 + 0.12 * std::sin(frame * 0.005), box * 1.6);
            scene.Clear(Hal::Rgb(0.01, 0.01, 0.03));

            for (int32_t i = 0; i < particleCount; i++)
            {
                double px = positions[static_cast<size_t>(4 * i)] * scale;
                double py = positions[static_cast<size_t>(4 * i + 1)] * scale;
                double pz = positions[static_cast<size_t>(4 * i + 2)] * scale;
                double speed = positions[static_cast<size_t>(4 * i + 3)];
                double t = std::min(1.0, speed / maxSpeed);
                scene.Sphere(px, py, pz, sphereRadius, SpeedColor(t));
            }

            bool presented = scene.Render();
            if (capture && presented)
            {
                if (scene.ReadCapturedFrame(lastFrame, lastWidth, lastHeight))
                {
                    if (!recorder)
                    {
                        recorder = std::make_unique<Hal::Examples::Mp4Recorder>("gpu_flow_field_3d.mp4", lastWidth, lastHeight, 30, 1);
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
            Hal::ExportPng("gpu_flow_field_3d.png", lastWidth, lastHeight, lastFrame);
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_flow_field_3d.mp4 and gpu_flow_field_3d.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuFlowField3D failed: ") + e.what());
        return 1;
    }
}
