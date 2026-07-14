// Showcase: luminous "ink in water" driven by the GPU implicit (ADI)
// diffusion solver. Three colored ink channels (red, green, blue) live on
// their own 320x320 toroidal fields; a ring of orbiting emitters injects
// glowing dye each frame, the fields bloom outward, and a gentle decay
// keeps the whole thing at a shimmering steady state instead of washing
// out to grey.
//
// The spreading is done by AdiKernel2D::RunSteps — the alternating-
// direction implicit solver whose tridiagonal systems are solved on the
// GPU (one workgroup per grid line, parallel cyclic reduction). Because
// ADI is unconditionally stable, each frame can take a few large-rate
// diffusion sub-steps that would blow up an explicit stencil, so the ink
// blooms dramatically per frame while the solve stays GPU-resident (one
// upload, N sweeps, one readback per channel). This is the implicit
// counterpart to the explicit reaction-diffusion showcase.
//
// Usage:
//   GpuInkFlow             live window
//   GpuInkFlow headless    no window; writes gpu_ink_flow.mp4 + a final png
//
// This is one of three GPU showcases (see also GpuReactionDiffusion,
// GpuSwarm).

#include "Hal/Color.h"
#include "Hal/Gpu/AdiKernel2D.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace
{
    // An orbiting dye source: it circles a center at its own radius and
    // angular speed, depositing a Gaussian blob tinted by (r, g, b).
    struct Emitter
    {
        double  centerX;
        double  centerY;
        double  orbitRadius;
        double  angularSpeed;
        double  phase;
        double  r;
        double  g;
        double  b;
    };

    // Deposit a Gaussian blob of the given color weights centered at
    // (cx, cy) into the three ink fields, stamped only within a local
    // window (a few sigma) so seeding stays cheap on the host.
    void Deposit(std::vector<float> &fr,
                 std::vector<float> &fg,
                 std::vector<float> &fb,
                 int32_t             nx,
                 int32_t             ny,
                 double              cx,
                 double              cy,
                 double              sigma,
                 double              amount,
                 double              wr,
                 double              wg,
                 double              wb)
    {
        int32_t reach = static_cast<int32_t>(sigma * 3.0) + 1;
        int32_t ix = static_cast<int32_t>(std::lround(cx));
        int32_t iy = static_cast<int32_t>(std::lround(cy));
        double inv2s2 = 1.0 / (2.0 * sigma * sigma);
        for (int32_t dx = -reach; dx <= reach; dx++)
        {
            for (int32_t dy = -reach; dy <= reach; dy++)
            {
                int32_t x = ((ix + dx) % nx + nx) % nx;
                int32_t y = ((iy + dy) % ny + ny) % ny;
                double d2 = static_cast<double>(dx * dx + dy * dy);
                double gauss = amount * std::exp(-d2 * inv2s2);
                size_t i = static_cast<size_t>(x * ny + y);
                fr[i] += static_cast<float>(gauss * wr);
                fg[i] += static_cast<float>(gauss * wg);
                fb[i] += static_cast<float>(gauss * wb);
            }
        }
    }

    // Filmic-ish tone map: turn an unbounded accumulated intensity into a
    // [0, 1] channel with a soft glowing shoulder rather than a hard clip.
    double ToneMap(double x)
    {
        if (x < 0.0)
        {
            x = 0.0;
        }
        return 1.0 - std::exp(-x);
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t nx = 320;
        const int32_t ny = 320;
        const int32_t visScale = 2;
        const int32_t substeps = 2;
        const float diffRate = 0.55f;
        const double decay = 0.985;
        const int32_t totalFrames = headless ? 360 : 100000;

        Hal::Gpu::ComputeContext context;
        // One implicit solver per color channel.
        Hal::Gpu::AdiKernel2D kernelR(context, nx, ny);
        Hal::Gpu::AdiKernel2D kernelG(context, nx, ny);
        Hal::Gpu::AdiKernel2D kernelB(context, nx, ny);

        std::vector<float> fr(static_cast<size_t>(nx * ny), 0.0f);
        std::vector<float> fg(static_cast<size_t>(nx * ny), 0.0f);
        std::vector<float> fb(static_cast<size_t>(nx * ny), 0.0f);

        // A ring of emitters in complementary colors, orbiting the center
        // at staggered speeds so the ink braids into a slow mandala.
        std::vector<Emitter> emitters;
        const int32_t emitterCount = 6;
        const double palette[6][3] = {
            {1.00, 0.20, 0.35},
            {1.00, 0.65, 0.10},
            {0.30, 1.00, 0.45},
            {0.15, 0.80, 1.00},
            {0.45, 0.30, 1.00},
            {1.00, 0.25, 0.85},
        };
        double cx = nx * 0.5;
        double cy = ny * 0.5;
        for (int32_t e = 0; e < emitterCount; e++)
        {
            Emitter em = {};
            em.centerX = cx;
            em.centerY = cy;
            em.orbitRadius = nx * 0.28;
            em.angularSpeed = 0.010 + 0.004 * e;
            em.phase = 2.0 * M_PI * e / emitterCount;
            em.r = palette[e][0];
            em.g = palette[e][1];
            em.b = palette[e][2];
            emitters.push_back(em);
        }

        Hal::Examples::GridWindow win("GPU Ink Flow (implicit ADI diffusion)", nx, ny, visScale, !headless);
        if (headless)
        {
            win.StartRecording("gpu_ink_flow.mp4", 30);
        }

        for (int32_t frame = 0; (frame < totalFrames) && !win.IsClosed(); frame++)
        {
            // Inject dye at each emitter's current orbital position.
            for (size_t e = 0; e < emitters.size(); e++)
            {
                Emitter &em = emitters[e];
                double angle = em.phase + em.angularSpeed * frame;
                // A slow radial pulse makes the ring breathe in and out.
                double radius = em.orbitRadius * (0.82 + 0.18 * std::sin(frame * 0.013 + em.phase));
                double px = em.centerX + radius * std::cos(angle);
                double py = em.centerY + radius * std::sin(angle);
                Deposit(fr, fg, fb, nx, ny, px, py, 3.2, 1.1, em.r, em.g, em.b);
            }

            // Spread each channel with a few large, unconditionally stable
            // implicit sub-steps, entirely on the GPU.
            kernelR.RunSteps(fr, substeps, diffRate, true, true);
            kernelG.RunSteps(fg, substeps, diffRate, true, true);
            kernelB.RunSteps(fb, substeps, diffRate, true, true);

            // Decay toward a luminous steady state and paint.
            for (int32_t x = 0; x < nx; x++)
            {
                for (int32_t y = 0; y < ny; y++)
                {
                    size_t i = static_cast<size_t>(x * ny + y);
                    fr[i] = static_cast<float>(fr[i] * decay);
                    fg[i] = static_cast<float>(fg[i] * decay);
                    fb[i] = static_cast<float>(fb[i] * decay);
                    double r = ToneMap(static_cast<double>(fr[i]) * 5.0);
                    double g = ToneMap(static_cast<double>(fg[i]) * 5.0);
                    double b = ToneMap(static_cast<double>(fb[i]) * 5.0);
                    win.SetPix(x, y, Hal::Rgb(r, g, b));
                }
            }
            win.TickPause(headless ? 0 : 1);
        }

        if (headless)
        {
            win.StopRecording();
            win.ToPng("gpu_ink_flow.png");
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_ink_flow.mp4 and gpu_ink_flow.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuInkFlow failed: ") + e.what());
        return 1;
    }
}
