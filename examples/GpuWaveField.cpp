// Showcase: a ripple tank. Several point sources oscillate on the surface
// of a 2D wave field and their circular wavefronts cross, building the
// shifting interference fringes and bright caustic lines you see where
// raindrops land on still water. The field is a real second-order wave
// equation (u_tt = c^2 * laplacian(u)) integrated by leapfrog on the GPU
// (WaveKernel2D) — oscillatory physics, the opposite of the diffusion the
// other showcases lean on: energy propagates and interferes instead of
// smoothing away.
//
// Each frame stamps the driven sources into the field, then advances a
// batch of GPU-resident leapfrog sub-steps. The wave amplitude (signed) is
// shaded through a diverging palette so crests and troughs read as warm and
// cool, with a bright rim on the steepest gradients for the caustic sheen.
//
// Usage:
//   GpuWaveField                live window (source pattern cycles)
//   GpuWaveField <pattern>      live window, one pattern
//                               (duet | orbit | grid | chaos)
//   GpuWaveField headless [pattern]
//                               no window; writes gpu_wave_field.mp4 + png
//
// One of the GPU showcases (see also GpuReactionDiffusion, GpuInkFlow,
// GpuSwarm, GpuSlimeMold, GpuAttractor3D).

#include "Hal/Color.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/WaveKernel2D.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace GpuWaveFieldInternal
{
    // A driven oscillator on the surface: position, temporal frequency,
    // and how it drifts over time (so the fringes never quite settle).
    struct Source
    {
        double baseX;
        double baseY;
        double frequency;
        double driftRadius;
        double driftSpeed;
        double phase;
    };

    // Diverging crest/trough palette: deep teal (trough) -> near-black
    // (rest) -> hot amber (crest), with white pushed onto the extremes for
    // the caustic highlight. s is the signed amplitude, roughly [-1, 1].
    int32_t WaveColor(double s)
    {
        if (s < -1.0)
        {
            s = -1.0;
        }
        if (s > 1.0)
        {
            s = 1.0;
        }
        double mag = std::abs(s);
        double r;
        double g;
        double b;
        if (s >= 0.0)
        {
            // Rest -> amber -> white.
            r = 0.05 + 0.95 * mag;
            g = 0.02 + 0.55 * mag * mag + 0.35 * std::pow(mag, 6.0);
            b = 0.06 + 0.60 * std::pow(mag, 8.0);
        }
        else
        {
            // Rest -> teal -> white.
            r = 0.05 + 0.20 * std::pow(mag, 6.0);
            g = 0.06 + 0.55 * mag;
            b = 0.10 + 0.75 * mag;
        }
        return Hal::Rgb(r, g, b);
    }

    // Stamp a soft Gaussian source value into the field (added, so multiple
    // drives superpose), within a local window.
    void DriveSource(std::vector<float> &cur,
                     int32_t             nx,
                     int32_t             ny,
                     double              cx,
                     double              cy,
                     double              value)
    {
        const double sigma = 2.2;
        int32_t reach = 6;
        int32_t ix = static_cast<int32_t>(std::lround(cx));
        int32_t iy = static_cast<int32_t>(std::lround(cy));
        double inv2s2 = 1.0 / (2.0 * sigma * sigma);
        for (int32_t dx = -reach; dx <= reach; dx++)
        {
            for (int32_t dy = -reach; dy <= reach; dy++)
            {
                int32_t x = ix + dx;
                int32_t y = iy + dy;
                if ((x < 0) || (x >= nx) || (y < 0) || (y >= ny))
                {
                    continue;
                }
                double falloff = std::exp(-static_cast<double>(dx * dx + dy * dy) * inv2s2);
                cur[static_cast<size_t>(x * ny + y)] += static_cast<float>(value * falloff);
            }
        }
    }

    // Build a named source pattern for the given grid.
    std::vector<Source> MakePattern(const std::string &name,
                                    int32_t            nx,
                                    int32_t            ny)
    {
        std::vector<Source> sources;
        double cx = nx * 0.5;
        double cy = ny * 0.5;

        if (name == "orbit")
        {
            for (int32_t i = 0; i < 3; i++)
            {
                Source s = {};
                s.baseX = cx;
                s.baseY = cy;
                s.frequency = 0.30 + 0.04 * i;
                s.driftRadius = nx * 0.30;
                s.driftSpeed = 0.010 + 0.004 * i;
                s.phase = 2.0 * M_PI * i / 3.0;
                sources.push_back(s);
            }
        }
        else if (name == "grid")
        {
            for (int32_t gx = 0; gx < 3; gx++)
            {
                for (int32_t gy = 0; gy < 3; gy++)
                {
                    Source s = {};
                    s.baseX = nx * (0.25 + 0.25 * gx);
                    s.baseY = ny * (0.25 + 0.25 * gy);
                    s.frequency = 0.34;
                    s.driftRadius = 0.0;
                    s.driftSpeed = 0.0;
                    s.phase = 0.0;
                    sources.push_back(s);
                }
            }
        }
        else if (name == "chaos")
        {
            const double positions[5][2] = {{0.30, 0.35}, {0.70, 0.30}, {0.50, 0.68}, {0.22, 0.70}, {0.78, 0.66}};
            for (int32_t i = 0; i < 5; i++)
            {
                Source s = {};
                s.baseX = nx * positions[i][0];
                s.baseY = ny * positions[i][1];
                s.frequency = 0.27 + 0.04 * i;
                s.driftRadius = nx * 0.06;
                s.driftSpeed = 0.013 + 0.006 * i;
                s.phase = 1.7 * i;
                sources.push_back(s);
            }
        }
        else
        {
            // "duet": the classic two-source interference demo.
            for (int32_t i = 0; i < 2; i++)
            {
                Source s = {};
                s.baseX = nx * (i == 0 ? 0.36 : 0.64);
                s.baseY = cy;
                s.frequency = 0.34;
                s.driftRadius = ny * 0.10;
                s.driftSpeed = 0.011;
                s.phase = (i == 0) ? 0.0 : M_PI * 0.15;
                sources.push_back(s);
            }
        }
        return sources;
    }

    const char *g_patterns[] = {"duet", "orbit", "grid", "chaos"};
    const int32_t g_patternCount = 4;
}

using namespace GpuWaveFieldInternal;

int main(int argc, char **argv)
{
    try
    {
        bool headless = false;
        std::string fixedPattern;
        for (int32_t a = 1; a < argc; a++)
        {
            if (std::strcmp(argv[a], "headless") == 0)
            {
                headless = true;
                continue;
            }
            for (int32_t p = 0; p < g_patternCount; p++)
            {
                if (std::strcmp(argv[a], g_patterns[p]) == 0)
                {
                    fixedPattern = g_patterns[p];
                }
            }
        }

        const int32_t nx = 400;
        const int32_t ny = 400;
        const int32_t visScale = 2;
        const int32_t substeps = 3;
        const float speed2 = 0.28f;
        const float damping = 0.9955f;
        const double driveAmp = 0.5;

        const int32_t framesPerPattern = headless ? 90 : 640;
        const int32_t totalFrames = (!fixedPattern.empty()) ? (headless ? 300 : 100000) : (framesPerPattern * g_patternCount);

        Hal::Gpu::ComputeContext context;
        Hal::Gpu::WaveKernel2D kernel(context, nx, ny);

        std::vector<float> cur(static_cast<size_t>(nx * ny), 0.0f);
        std::vector<float> prev(static_cast<size_t>(nx * ny), 0.0f);

        std::string activeName = fixedPattern.empty() ? g_patterns[0] : fixedPattern;
        std::vector<Source> sources = MakePattern(activeName, nx, ny);

        Hal::Examples::GridWindow win("GPU Wave Field (ripple-tank interference)", nx, ny, visScale, !headless);
        if (headless)
        {
            win.StartRecording("gpu_wave_field.mp4", 30);
        }

        for (int32_t frame = 0; (frame < totalFrames) && !win.IsClosed(); frame++)
        {
            if (fixedPattern.empty())
            {
                std::string wanted = g_patterns[(frame / framesPerPattern) % g_patternCount];
                if (wanted != activeName)
                {
                    activeName = wanted;
                    sources = MakePattern(activeName, nx, ny);
                    std::fill(cur.begin(), cur.end(), 0.0f);
                    std::fill(prev.begin(), prev.end(), 0.0f);
                }
            }

            // Drive each source at its current drifted position.
            for (size_t s = 0; s < sources.size(); s++)
            {
                const Source &src = sources[s];
                double angle = src.phase + src.driftSpeed * frame;
                double px = src.baseX + src.driftRadius * std::cos(angle);
                double py = src.baseY + src.driftRadius * std::sin(angle);
                double value = driveAmp * std::sin(src.frequency * frame + src.phase);
                DriveSource(cur, nx, ny, px, py, value);
            }

            kernel.RunSteps(cur, prev, substeps, speed2, damping, false);

            for (int32_t x = 0; x < nx; x++)
            {
                for (int32_t y = 0; y < ny; y++)
                {
                    // Amplify small amplitudes so faint far-field ripples
                    // stay visible without clipping the near field.
                    double s = std::tanh(static_cast<double>(cur[static_cast<size_t>(x * ny + y)]) * 1.3);
                    win.SetPix(x, y, WaveColor(s));
                }
            }
            win.TickPause(headless ? 0 : 1);
        }

        if (headless)
        {
            win.StopRecording();
            win.ToPng("gpu_wave_field.png");
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_wave_field.mp4 and gpu_wave_field.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuWaveField failed: ") + e.what());
        return 1;
    }
}
