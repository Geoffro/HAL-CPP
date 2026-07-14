// Showcase: a strange attractor rendered as a living 3D point cloud on the
// hardware Vulkan renderer (Hal::Gpu::Scene3DRenderContext — real depth
// testing, instanced sphere meshes, Lambert lighting). Thousands of
// particles are dropped onto a chaotic flow and integrated forward every
// frame; because the flow is sensitive to initial conditions they smear
// across the whole attractor, tracing out its shape — the Lorenz butterfly,
// Aizawa's cored sphere, Thomas's cyclically-symmetric web, Halvorsen's
// three-lobed knot. Each particle is tinted by its speed, and the camera
// auto-orbits so the structure turns in space.
//
// Unlike the 2D showcases this one is the GPU *rendering* path, not GPU
// compute (the flow integration is on the host). It always needs a live
// window (that window is the Vulkan surface), but it can still record a
// clip: pass a frame count and it enables the renderer's opt-in frame
// capture (Scene3DRenderContext::SetCaptureEnabled — a per-frame readback,
// so it is only turned on for a bounded capture run, never a free-running
// interactive one) and writes gpu_attractor3d.mp4 + a final .png.
//
// The attractor is auto-fit: a probe trajectory is measured at startup and
// the whole cloud is scaled/centered into the view box, so every attractor
// frames itself correctly regardless of its natural size.
//
// Usage:
//   GpuAttractor3D                 live window, Lorenz (runs until closed)
//   GpuAttractor3D <name>          lorenz | aizawa | thomas | halvorsen
//   GpuAttractor3D <name> <frames> run <frames> frames, recording an mp4+png
//
// One of the GPU showcases (see also GpuReactionDiffusion, GpuInkFlow,
// GpuSwarm, GpuSlimeMold, GpuWaveField).

#include "Hal/Color.h"
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
    struct Vec3d
    {
        double x;
        double y;
        double z;
    };

    // Evaluate the chosen attractor's velocity field at p. `which` selects
    // the system; parameters are the standard textbook values.
    Vec3d Velocity(int32_t       which,
                   const Vec3d  &p)
    {
        Vec3d v = {};
        if (which == 1)
        {
            // Aizawa.
            const double a = 0.95;
            const double b = 0.7;
            const double c = 0.6;
            const double d = 3.5;
            const double e = 0.25;
            const double f = 0.1;
            v.x = (p.z - b) * p.x - d * p.y;
            v.y = d * p.x + (p.z - b) * p.y;
            v.z = c + a * p.z - (p.z * p.z * p.z) / 3.0 - (p.x * p.x + p.y * p.y) * (1.0 + e * p.z) + f * p.z * p.x * p.x * p.x;
        }
        else if (which == 2)
        {
            // Thomas (cyclically symmetric).
            const double b = 0.208;
            v.x = std::sin(p.y) - b * p.x;
            v.y = std::sin(p.z) - b * p.y;
            v.z = std::sin(p.x) - b * p.z;
        }
        else if (which == 3)
        {
            // Halvorsen.
            const double a = 1.4;
            v.x = -a * p.x - 4.0 * p.y - 4.0 * p.z - p.y * p.y;
            v.y = -a * p.y - 4.0 * p.z - 4.0 * p.x - p.z * p.z;
            v.z = -a * p.z - 4.0 * p.x - 4.0 * p.y - p.x * p.x;
        }
        else
        {
            // Lorenz.
            const double sigma = 10.0;
            const double rho = 28.0;
            const double beta = 8.0 / 3.0;
            v.x = sigma * (p.y - p.x);
            v.y = p.x * (rho - p.z) - p.y;
            v.z = p.x * p.y - beta * p.z;
        }
        return v;
    }

    // Per-attractor integration step and seed, tuned so the flow stays
    // stable at the chosen dt.
    struct AttractorConfig
    {
        int32_t which;
        double  dt;
        Vec3d   seed;
    };

    AttractorConfig ConfigFor(const std::string &name)
    {
        AttractorConfig cfg = {};
        if (name == "aizawa")
        {
            cfg.which = 1;
            cfg.dt = 0.008;
            cfg.seed = {0.1, 0.0, 0.0};
        }
        else if (name == "thomas")
        {
            cfg.which = 2;
            cfg.dt = 0.015;
            cfg.seed = {0.1, 0.0, 0.0};
        }
        else if (name == "halvorsen")
        {
            cfg.which = 3;
            cfg.dt = 0.004;
            cfg.seed = {-1.0, 0.0, 0.0};
        }
        else
        {
            cfg.which = 0;
            cfg.dt = 0.005;
            cfg.seed = {0.1, 0.0, 0.0};
        }
        return cfg;
    }

    Vec3d Step(int32_t      which,
               const Vec3d &p,
               double       dt)
    {
        // Midpoint (RK2) — steadier than Euler for the stiffer systems.
        Vec3d k1 = Velocity(which, p);
        Vec3d mid = {p.x + 0.5 * dt * k1.x, p.y + 0.5 * dt * k1.y, p.z + 0.5 * dt * k1.z};
        Vec3d k2 = Velocity(which, mid);
        Vec3d out = {p.x + dt * k2.x, p.y + dt * k2.y, p.z + dt * k2.z};
        return out;
    }

    // Magma-ish speed palette: dark plum -> magenta -> orange -> white.
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
            {0.05, 0.02, 0.20},
            {0.45, 0.05, 0.45},
            {0.85, 0.20, 0.35},
            {0.99, 0.62, 0.20},
            {1.00, 0.98, 0.85},
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

int main(int argc, char **argv)
{
    try
    {
        std::string name = "lorenz";
        int32_t maxFrames = 0; // 0 == run until the window is closed.
        for (int32_t a = 1; a < argc; a++)
        {
            int32_t parsed = std::atoi(argv[a]);
            if (parsed > 0)
            {
                maxFrames = parsed;
            }
            else
            {
                name = argv[a];
            }
        }

        AttractorConfig cfg = ConfigFor(name);

        // Probe one trajectory to measure the attractor's extent (after a
        // burn-in onto the attractor), so the cloud can be auto-fit into
        // the view box regardless of the system's natural scale.
        Vec3d probe = cfg.seed;
        for (int32_t i = 0; i < 3000; i++)
        {
            probe = Step(cfg.which, probe, cfg.dt);
        }
        // Seed the particles ALONG a long trajectory so the cloud traces
        // the whole attractor from the very first frame — a tight initial
        // cluster would take far too long to disperse — while measuring the
        // extent + peak speed for the auto-fit and the speed colouring.
        Hal::Rand rand;
        const int32_t particleCount = 9000;
        const int32_t collectStride = 3;
        std::vector<Vec3d> particles(static_cast<size_t>(particleCount));
        Vec3d lo = probe;
        Vec3d hi = probe;
        double maxSpeed = 1e-9;
        for (int32_t i = 0; i < particleCount; i++)
        {
            for (int32_t s = 0; s < collectStride; s++)
            {
                probe = Step(cfg.which, probe, cfg.dt);
            }
            lo.x = std::min(lo.x, probe.x);
            lo.y = std::min(lo.y, probe.y);
            lo.z = std::min(lo.z, probe.z);
            hi.x = std::max(hi.x, probe.x);
            hi.y = std::max(hi.y, probe.y);
            hi.z = std::max(hi.z, probe.z);
            Vec3d vel = Velocity(cfg.which, probe);
            maxSpeed = std::max(maxSpeed, std::sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z));
            particles[static_cast<size_t>(i)] = probe;
        }

        Vec3d attractorCenter = {0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y), 0.5 * (lo.z + hi.z)};
        double extent = std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z));
        if (extent < 1e-6)
        {
            extent = 1.0;
        }

        const double box = 40.0;
        const double center = box / 2.0;
        double scale = (box * 0.85) / extent;
        double sphereRadius = box * 0.0045;

        // A little jitter off the curve (scaled to the attractor's size)
        // gives the traced cloud some volume instead of a hairline.
        double jitter = extent * 0.006;
        for (int32_t i = 0; i < particleCount; i++)
        {
            particles[static_cast<size_t>(i)].x += rand.Gaussian(0.0, jitter);
            particles[static_cast<size_t>(i)].y += rand.Gaussian(0.0, jitter);
            particles[static_cast<size_t>(i)].z += rand.Gaussian(0.0, jitter);
        }

        Hal::Gpu::Window               window(960, 720, "GPU Attractor 3D (" + name + ")");
        Hal::Gpu::Scene3DRenderContext scene(window, box, box, box);
        scene.SetCamera(0.6, 0.35, box * 1.35);

        // A bounded run (frame count given) records a clip; the readback is
        // only worth its per-frame stall for capture, so leave it off for a
        // free-running interactive session.
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
        while (!window.ShouldClose())
        {
            window.PollEvents();

            // Auto-orbit the camera.
            yaw += 0.004;
            scene.SetCamera(yaw, 0.35 + 0.12 * std::sin(frame * 0.004), box * 1.35);
            scene.Clear(Hal::Rgb(0.02, 0.02, 0.05));

            // A few integration sub-steps per frame for smooth, fast flow.
            for (int32_t sub = 0; sub < 4; sub++)
            {
                for (int32_t i = 0; i < particleCount; i++)
                {
                    particles[static_cast<size_t>(i)] = Step(cfg.which, particles[static_cast<size_t>(i)], cfg.dt);
                }
            }

            for (int32_t i = 0; i < particleCount; i++)
            {
                const Vec3d &p = particles[static_cast<size_t>(i)];
                Vec3d vel = Velocity(cfg.which, p);
                double speed = std::sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
                double t = std::tanh(speed / maxSpeed * 1.5);

                double wx = center + (p.x - attractorCenter.x) * scale;
                double wy = center + (p.y - attractorCenter.y) * scale;
                double wz = center + (p.z - attractorCenter.z) * scale;
                scene.Sphere(wx, wy, wz, sphereRadius, SpeedColor(t));
            }

            bool presented = scene.Render();

            if (capture && presented)
            {
                if (scene.ReadCapturedFrame(lastFrame, lastWidth, lastHeight))
                {
                    if (!recorder)
                    {
                        recorder = std::make_unique<Hal::Examples::Mp4Recorder>("gpu_attractor3d.mp4", lastWidth, lastHeight, 30, 1);
                    }
                    recorder->AddFrame(lastFrame);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(16));

            frame++;
            if ((maxFrames > 0) && (frame >= maxFrames))
            {
                break;
            }
        }

        if (capture && !lastFrame.empty())
        {
            recorder.reset();
            Hal::ExportPng("gpu_attractor3d.png", lastWidth, lastHeight, lastFrame);
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_attractor3d.mp4 and gpu_attractor3d.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuAttractor3D failed: ") + e.what());
        return 1;
    }
}
