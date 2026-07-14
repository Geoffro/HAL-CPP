// Showcase: a living swarm of 150,000 particles, choreographed on the GPU.
// Every particle repels its neighbors through the same bucketed neighbor
// search + linear-spring force the agent-based models use
// (AgentForceKernel2D), so the cloud has real internal spacing and texture
// instead of collapsing to a point; on top of that the host adds a few
// slowly orbiting attractors that hand the particles back and forth,
// braiding them into flowing, breathing filaments.
//
// The expensive part — every particle finding its neighbors and summing
// their forces, an O(N^2) computation done in O(N) with a spatial hash —
// runs on the GPU each frame. At 150k particles that neighbor solve is
// where the GPU decisively beats the CPU (see the agent-force rows of the
// backend benchmark); the host only integrates positions and paints.
// Particle trails are accumulated into a persistence buffer so motion
// reads as light.
//
// Usage:
//   GpuSwarm             live window
//   GpuSwarm headless    no window; writes gpu_swarm.mp4 + a final png
//   GpuSwarm <N>         override the particle count (also works headless)
//
// This is one of three GPU showcases (see also GpuReactionDiffusion,
// GpuInkFlow).

#include "Hal/Color.h"
#include "Hal/Gpu/AgentForceKernel2D.h"
#include "Hal/Gpu/ComputeContext.h"
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

namespace
{
    // A slowly orbiting point of gravity the particles are drawn toward.
    struct Attractor
    {
        double centerX;
        double centerY;
        double orbitRadius;
        double angularSpeed;
        double phase;
    };

    // Plasma palette for the accumulated particle density: black -> indigo
    // -> magenta -> hot orange -> white, so dense filaments read as light.
    int32_t PlasmaColor(double t)
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
            {0.00, 0.00, 0.00},
            {0.17, 0.02, 0.42},
            {0.72, 0.12, 0.62},
            {0.99, 0.62, 0.20},
            {1.00, 1.00, 0.96},
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
        bool headless = false;
        int32_t requestedCount = 150000;
        for (int32_t a = 1; a < argc; a++)
        {
            if (std::strcmp(argv[a], "headless") == 0)
            {
                headless = true;
                continue;
            }
            int32_t parsed = std::atoi(argv[a]);
            if (parsed > 0)
            {
                requestedCount = parsed;
            }
        }

        const int32_t numAgents = requestedCount;
        const int32_t nx = 420;
        const int32_t ny = 420;
        const int32_t visScale = 2;
        const int32_t totalFrames = headless ? 360 : 100000;

        // Force / integration constants. The attractor pull is strong
        // enough to gather the particles into a few dense, moving knots;
        // the GPU neighbor repulsion then gives those knots finite size
        // and internal texture (a pressure that stops them collapsing to a
        // point), and the particles handed between attractors trail into
        // bright filaments.
        const float interactionRad = 1.6f;
        const float springConstant = 1.9f;
        const double dt = 0.5;
        const double damping = 0.905;
        const double repelScale = 1.0;
        const double attractStrength = 1.25;
        const double attractFalloff = 20.0;
        const double trailDecay = 0.66;

        Hal::Gpu::ComputeContext context;
        Hal::Gpu::AgentForceKernel2D kernel(context, nx, ny, numAgents);
        Hal::Rand rand;

        std::vector<float> positions(static_cast<size_t>(numAgents) * 2);
        std::vector<float> velocities(static_cast<size_t>(numAgents) * 2, 0.0f);
        std::vector<float> radii(static_cast<size_t>(numAgents), 0.8f);
        for (int32_t i = 0; i < numAgents; i++)
        {
            positions[static_cast<size_t>(2 * i)] = static_cast<float>(rand.Double(nx));
            positions[static_cast<size_t>(2 * i + 1)] = static_cast<float>(rand.Double(ny));
        }

        // Three attractors orbiting the center at staggered radii/speeds.
        std::vector<Attractor> attractors;
        const int32_t attractorCount = 3;
        for (int32_t a = 0; a < attractorCount; a++)
        {
            Attractor at = {};
            at.centerX = nx * 0.5;
            at.centerY = ny * 0.5;
            at.orbitRadius = nx * (0.22 + 0.09 * a);
            at.angularSpeed = 0.011 + 0.006 * a;
            at.phase = 2.0 * M_PI * a / attractorCount;
            attractors.push_back(at);
        }

        std::vector<float> intensity(static_cast<size_t>(nx * ny), 0.0f);

        Hal::Examples::GridWindow win("GPU Swarm (150k agent-force particles)", nx, ny, visScale, !headless);
        if (headless)
        {
            win.StartRecording("gpu_swarm.mp4", 30);
        }

        for (int32_t frame = 0; (frame < totalFrames) && !win.IsClosed(); frame++)
        {
            // GPU: neighbor repulsion forces for every particle.
            kernel.UploadAgents(positions, radii, numAgents);
            kernel.Dispatch(interactionRad, springConstant, true, true);
            std::vector<float> forces = kernel.DownloadVelocityDeltas();

            // Current attractor positions.
            double ax[8];
            double ay[8];
            for (int32_t a = 0; a < attractorCount; a++)
            {
                double angle = attractors[a].phase + attractors[a].angularSpeed * frame;
                ax[a] = attractors[a].centerX + attractors[a].orbitRadius * std::cos(angle);
                ay[a] = attractors[a].centerY + attractors[a].orbitRadius * std::sin(angle);
            }

            // Host: integrate. Repulsion (GPU) + attractor pull + damping.
            for (int32_t i = 0; i < numAgents; i++)
            {
                size_t xi = static_cast<size_t>(2 * i);
                size_t yi = xi + 1;
                double px = positions[xi];
                double py = positions[yi];
                double vx = velocities[xi];
                double vy = velocities[yi];

                vx += forces[xi] * repelScale;
                vy += forces[yi] * repelScale;

                // Pull toward each attractor as a unit direction weighted
                // by proximity (falloff/(dist+falloff)), so the nearest one
                // dominates but the others still tug — particles get handed
                // between attractors and stream along filaments.
                for (int32_t a = 0; a < attractorCount; a++)
                {
                    double dx = ax[a] - px;
                    double dy = ay[a] - py;
                    double dist = std::sqrt(dx * dx + dy * dy) + 1.0;
                    double weight = attractFalloff / (dist + attractFalloff);
                    double pull = attractStrength * weight / dist;
                    vx += dx * pull;
                    vy += dy * pull;
                }

                vx *= damping;
                vy *= damping;

                px += vx * dt;
                py += vy * dt;

                // Toroidal wrap to match the kernel's neighbor search.
                px = std::fmod(px + nx, static_cast<double>(nx));
                py = std::fmod(py + ny, static_cast<double>(ny));

                positions[xi] = static_cast<float>(px);
                positions[yi] = static_cast<float>(py);
                velocities[xi] = static_cast<float>(vx);
                velocities[yi] = static_cast<float>(vy);
            }

            // Accumulate particle density into the persistence buffer.
            for (size_t i = 0; i < intensity.size(); i++)
            {
                intensity[i] = static_cast<float>(intensity[i] * trailDecay);
            }
            // Soft 3x3 splat (center-weighted) so the density reads as a
            // smooth glow rather than per-pixel Poisson noise.
            for (int32_t i = 0; i < numAgents; i++)
            {
                int32_t x = static_cast<int32_t>(positions[static_cast<size_t>(2 * i)]);
                int32_t y = static_cast<int32_t>(positions[static_cast<size_t>(2 * i + 1)]);
                for (int32_t dx = -1; dx <= 1; dx++)
                {
                    for (int32_t dy = -1; dy <= 1; dy++)
                    {
                        int32_t xx = x + dx;
                        int32_t yy = y + dy;
                        if ((xx >= 0) && (xx < nx) && (yy >= 0) && (yy < ny))
                        {
                            float w = ((dx == 0) && (dy == 0)) ? 0.5f : (((dx == 0) || (dy == 0)) ? 0.18f : 0.08f);
                            intensity[static_cast<size_t>(xx * ny + yy)] += w;
                        }
                    }
                }
            }

            for (int32_t x = 0; x < nx; x++)
            {
                for (int32_t y = 0; y < ny; y++)
                {
                    double t = 1.0 - std::exp(-static_cast<double>(intensity[static_cast<size_t>(x * ny + y)]) * 0.14);
                    win.SetPix(x, y, PlasmaColor(t));
                }
            }
            win.TickPause(headless ? 0 : 1);
        }

        if (headless)
        {
            win.StopRecording();
            win.ToPng("gpu_swarm.png");
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_swarm.mp4 and gpu_swarm.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuSwarm failed: ") + e.what());
        return 1;
    }
}
