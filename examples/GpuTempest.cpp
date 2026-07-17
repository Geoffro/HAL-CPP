// Showcase: a lightning storm over a wild sea. The "sea" is a real 2D wave
// equation (WaveKernel2D, leapfrog on the GPU) driven not by a fixed preset
// but by point sources that are born and die at random — random position,
// random pitch, random lifespan, unseeded RNG — so no two runs, and no two
// stretches of the same run, ever interfere the same way. 150,000 motes
// ride that turbulent surface: AgentForceKernel2D keeps them spaced on the
// GPU, and every agent climbs the local wave gradient on the host each
// frame, so the cloud is dragged into the field's ridges and troughs
// instead of drifting independently of it — the PDE steers the agents, the
// reverse of the deposit-driven hybrids (GpuSlimeMold, GpuPlasmaVortex).
// Whenever a mote's local wave amplitude snaps past a threshold — a
// wavefront sweeping under it — it fires: a bright white flash that decays
// over a few frames, so crossing wavefronts read as chains of lightning.
//
// This is a three-kernel hybrid, as GPU-intensive as the showcases get:
// WaveKernel2D advances the sea, AgentForceKernel2D spaces the motes, and
// DiffusionKernel2D blooms the three-channel flash-color trail they paint
// — all three GPU-resident, every frame.
//
// Usage:
//   GpuTempest             live window
//   GpuTempest headless    no window; writes gpu_tempest.mp4 + a final png
//   GpuTempest <N>         override the mote count (also works headless)
//
// One of the GPU showcases (see also GpuWaveField, GpuPlasmaVortex,
// GpuSlimeMold).

#include "Hal/Color.h"
#include "Hal/Gpu/AgentForceKernel2D.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/DiffusionKernel2D.h"
#include "Hal/Gpu/WaveKernel2D.h"
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

namespace GpuTempestInternal
{
    // One driven point on the wave surface, alive for [birthFrame,
    // birthFrame + lifespanFrames). Its amplitude ramps in and back out
    // (see EnvelopeAt) instead of switching on/off, so births and deaths
    // never pop.
    struct Source
    {
        double  x;
        double  y;
        double  frequency;
        double  amplitude;
        double  phase;
        int32_t birthFrame;
        int32_t lifespanFrames;
    };

    // Smoothstep fade-in over the first 12% of the source's life and
    // fade-out over the last 18%, so the two windows overlap into one
    // hump for very short-lived sources instead of ever going negative.
    double EnvelopeAt(int32_t age,
                      int32_t lifespanFrames)
    {
        double frac = static_cast<double>(age) / static_cast<double>(lifespanFrames);
        double fadeIn = frac / 0.12;
        if (fadeIn < 0.0)
        {
            fadeIn = 0.0;
        }
        if (fadeIn > 1.0)
        {
            fadeIn = 1.0;
        }
        fadeIn = fadeIn * fadeIn * (3.0 - 2.0 * fadeIn);
        double fadeOut = (1.0 - frac) / 0.18;
        if (fadeOut < 0.0)
        {
            fadeOut = 0.0;
        }
        if (fadeOut > 1.0)
        {
            fadeOut = 1.0;
        }
        fadeOut = fadeOut * fadeOut * (3.0 - 2.0 * fadeOut);
        return fadeIn * fadeOut;
    }

    // Stamp a soft Gaussian source value into the field (added, so
    // multiple drives superpose), within a local window.
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
                int32_t x = ((ix + dx) % nx + nx) % nx;
                int32_t y = ((iy + dy) % ny + ny) % ny;
                double falloff = std::exp(-static_cast<double>(dx * dx + dy * dy) * inv2s2);
                cur[static_cast<size_t>(x * ny + y)] += static_cast<float>(value * falloff);
            }
        }
    }

    // Nearest-cell field sample with toroidal wrap.
    float SampleField(const std::vector<float> &field,
                      int32_t                   nx,
                      int32_t                   ny,
                      double                    x,
                      double                    y)
    {
        int32_t ix = ((static_cast<int32_t>(std::lround(x)) % nx) + nx) % nx;
        int32_t iy = ((static_cast<int32_t>(std::lround(y)) % ny) + ny) % ny;
        return field[static_cast<size_t>(ix * ny + iy)];
    }

    // Bold diverging palette for the (tanh-compressed) signed amplitude:
    // crest runs deep red -> hot magenta -> white, trough runs deep blue
    // -> electric cyan -> white — far more saturated than a natural
    // ripple-tank shading, for an "extremely vibrant" storm rather than a
    // calm one. Returned as floats (not packed) so callers can blend a
    // flash before final tone mapping.
    void FieldColor(double  s,
                    double &outR,
                    double &outG,
                    double &outB)
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
        if (s >= 0.0)
        {
            outR = 0.08 + 0.92 * mag;
            outG = 0.02 * mag + 0.85 * std::pow(mag, 5.0);
            outB = 0.10 * mag + 0.90 * std::pow(mag, 7.0);
        }
        else
        {
            outR = 0.05 * mag + 0.55 * std::pow(mag, 6.0);
            outG = 0.10 + 0.55 * mag;
            outB = 0.15 + 0.90 * mag;
        }
    }
}

using namespace GpuTempestInternal;

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
        const int32_t nx = 440;
        const int32_t ny = 440;
        const int32_t visScale = 2;
        const int32_t totalFrames = headless ? 360 : 100000;

        // Wave equation (GPU WaveKernel2D), same stable shape as
        // GpuWaveField.
        const int32_t waveSubsteps = 3;
        const float   speed2 = 0.28f;
        const float   waveDamping = 0.9955f;
        const double  driveAmpScale = 0.7;

        // Random source lifecycle — no preset pattern, no fixed seed:
        // Hal::Rand() draws from entropy, so the sequence of births
        // below differs every run.
        const int32_t maxSources = 7;
        const double  spawnChance = 0.028;
        const int32_t minLifespan = 200;
        const int32_t maxLifespanExtra = 260;

        // Mote repulsion (GPU AgentForceKernel2D), same shape as
        // GpuSwarm.
        const float  interactionRad = 1.4f;
        const float  springConstant = 1.5f;
        const double dt = 0.5;
        const double moteDamping = 0.90;
        const double repelScale = 1.0;

        // How hard a mote climbs the local wave gradient, and how fast
        // that climb is capped so a steep wavefront can't fling it clear
        // off the field in one step.
        const double climbStrength = 5.5;
        const double maxSpeed = 6.0;

        // A mote fires when |field| crosses this threshold on a rising
        // edge; the flash decays fast, reading as a snap of lightning
        // rather than a slow fade.
        const double flashThreshold = 0.32;
        const float  flashDecay = 0.72f;

        // Flash-trail deposit / bloom.
        const float  depositAmount = 1.3f;
        const float  diffRate = 0.22f;
        const int32_t diffusionSubSteps = 2;
        const double trailDecay = 0.80;
        const double exposure = 0.16;
        const double saturationBoost = 1.5;
        const double backgroundWeight = 0.4;

        Hal::Gpu::ComputeContext context;
        Hal::Gpu::WaveKernel2D waveKernel(context, nx, ny);
        Hal::Gpu::AgentForceKernel2D forceKernel(context, nx, ny, numAgents);
        Hal::Gpu::DiffusionKernel2D diffusionKernel(context, nx, ny);
        Hal::Rand rand;

        std::vector<float> cur(static_cast<size_t>(nx * ny), 0.0f);
        std::vector<float> prev(static_cast<size_t>(nx * ny), 0.0f);
        std::vector<Source> sources;

        std::vector<float> positions(static_cast<size_t>(numAgents) * 2);
        std::vector<float> velocities(static_cast<size_t>(numAgents) * 2, 0.0f);
        std::vector<float> radii(static_cast<size_t>(numAgents), 0.7f);
        std::vector<float> flashLevel(static_cast<size_t>(numAgents), 0.0f);
        std::vector<float> prevFieldMag(static_cast<size_t>(numAgents), 0.0f);
        for (int32_t i = 0; i < numAgents; i++)
        {
            positions[static_cast<size_t>(2 * i)]     = static_cast<float>(rand.Double(nx));
            positions[static_cast<size_t>(2 * i + 1)] = static_cast<float>(rand.Double(ny));
        }

        std::vector<float> fr(static_cast<size_t>(nx * ny), 0.0f);
        std::vector<float> fg(static_cast<size_t>(nx * ny), 0.0f);
        std::vector<float> fb(static_cast<size_t>(nx * ny), 0.0f);

        Hal::Examples::GridWindow win("GPU Tempest (chaotic wave field + reactive agents)", nx, ny, visScale, !headless);
        if (headless)
        {
            win.StartRecording("gpu_tempest.mp4", 30);
        }

        for (int32_t frame = 0; (frame < totalFrames) && !win.IsClosed(); frame++)
        {
            // Randomly spawn a new source (bounded pool), independent of
            // any wall-clock or frame-count schedule — genuinely
            // unpredictable timing.
            if ((static_cast<int32_t>(sources.size()) < maxSources) && (rand.Double() < spawnChance))
            {
                Source s = {};
                s.x = nx * (0.12 + 0.76 * rand.Double());
                s.y = ny * (0.12 + 0.76 * rand.Double());
                s.frequency = 0.20 + 0.22 * rand.Double();
                s.amplitude = 0.5 + 0.5 * rand.Double();
                s.phase = 2.0 * M_PI * rand.Double();
                s.birthFrame = frame;
                s.lifespanFrames = minLifespan + rand.Int(maxLifespanExtra);
                sources.push_back(s);
            }

            // Drive every live source, dropping the ones that have aged
            // out.
            std::vector<Source> survivors;
            for (size_t s = 0; s < sources.size(); s++)
            {
                int32_t age = frame - sources[s].birthFrame;
                if (age > sources[s].lifespanFrames)
                {
                    continue;
                }
                double envelope = EnvelopeAt(age, sources[s].lifespanFrames);
                double value = driveAmpScale * sources[s].amplitude * envelope * std::sin(sources[s].frequency * age + sources[s].phase);
                DriveSource(cur, nx, ny, sources[s].x, sources[s].y, value);
                survivors.push_back(sources[s]);
            }
            sources = survivors;

            // GPU: propagate the wave equation.
            waveKernel.RunSteps(cur, prev, waveSubsteps, speed2, waveDamping, true);

            // GPU: mote neighbor repulsion.
            forceKernel.UploadAgents(positions, radii, numAgents);
            forceKernel.Dispatch(interactionRad, springConstant, true, true);
            std::vector<float> forces = forceKernel.DownloadVelocityDeltas();

            // Host: each mote climbs the local wave gradient, gets
            // shoved by its neighbors, and fires a flash on a rising
            // amplitude edge.
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

                float here = SampleField(cur, nx, ny, px, py);
                float right = SampleField(cur, nx, ny, px + 1.0, py);
                float left = SampleField(cur, nx, ny, px - 1.0, py);
                float up = SampleField(cur, nx, ny, px, py + 1.0);
                float down = SampleField(cur, nx, ny, px, py - 1.0);
                double gx = (static_cast<double>(right) - static_cast<double>(left)) * 0.5;
                double gy = (static_cast<double>(up) - static_cast<double>(down)) * 0.5;
                vx += gx * climbStrength;
                vy += gy * climbStrength;

                double fieldMag = std::abs(static_cast<double>(here));
                size_t si = static_cast<size_t>(i);
                if ((fieldMag >= flashThreshold) && (prevFieldMag[si] < flashThreshold))
                {
                    flashLevel[si] = 1.0f;
                }
                else
                {
                    flashLevel[si] = flashLevel[si] * flashDecay;
                }
                prevFieldMag[si] = static_cast<float>(fieldMag);

                vx *= moteDamping;
                vy *= moteDamping;
                double speed = std::sqrt(vx * vx + vy * vy);
                if (speed > maxSpeed)
                {
                    double scale = maxSpeed / speed;
                    vx *= scale;
                    vy *= scale;
                }

                px += vx * dt;
                py += vy * dt;
                px = std::fmod(px + nx, static_cast<double>(nx));
                py = std::fmod(py + ny, static_cast<double>(ny));

                positions[xi] = static_cast<float>(px);
                positions[yi] = static_cast<float>(py);
                velocities[xi] = static_cast<float>(vx);
                velocities[yi] = static_cast<float>(vy);
            }

            // Bleed each mote's color into the three trail fields with a
            // soft 3x3 splat. Motes glow warm amber — deliberately not
            // the background's red/blue field hue — so the agent layer
            // reads as its own thing riding the sea rather than blending
            // into it; brightness still tracks how strong the local wave
            // is, and a flash burns the amber up to pure white.
            for (int32_t i = 0; i < numAgents; i++)
            {
                size_t si = static_cast<size_t>(i);
                double s = std::tanh(static_cast<double>(SampleField(cur, nx, ny, positions[static_cast<size_t>(2 * i)], positions[static_cast<size_t>(2 * i + 1)])) * 1.3);
                double flash = flashLevel[si];
                double cr = 1.00;
                double cg = 0.62 + (1.0 - 0.62) * flash;
                double cb = 0.12 + (1.0 - 0.12) * flash;

                double amount = depositAmount * (0.30 + 0.70 * std::abs(s)) * (1.0 + 2.2 * flash);
                float tr = static_cast<float>(cr * amount);
                float tg = static_cast<float>(cg * amount);
                float tb = static_cast<float>(cb * amount);

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
                            size_t cell = static_cast<size_t>(xx * ny + yy);
                            fr[cell] += tr * w;
                            fg[cell] += tg * w;
                            fb[cell] += tb * w;
                        }
                    }
                }
            }

            // GPU: bloom and fade all three flash-trail channels.
            diffusionKernel.RunSteps(fr, diffusionSubSteps, diffRate, true, true);
            diffusionKernel.RunSteps(fg, diffusionSubSteps, diffRate, true, true);
            diffusionKernel.RunSteps(fb, diffusionSubSteps, diffRate, true, true);

            for (int32_t x = 0; x < nx; x++)
            {
                for (int32_t y = 0; y < ny; y++)
                {
                    size_t i = static_cast<size_t>(x * ny + y);
                    fr[i] = static_cast<float>(fr[i] * trailDecay);
                    fg[i] = static_cast<float>(fg[i] * trailDecay);
                    fb[i] = static_cast<float>(fb[i] * trailDecay);

                    double bgS = std::tanh(static_cast<double>(cur[i]) * 1.3);
                    double bgR;
                    double bgG;
                    double bgB;
                    FieldColor(bgS, bgR, bgG, bgB);

                    double r = bgR * backgroundWeight + (1.0 - std::exp(-static_cast<double>(fr[i]) * exposure));
                    double g = bgG * backgroundWeight + (1.0 - std::exp(-static_cast<double>(fg[i]) * exposure));
                    double b = bgB * backgroundWeight + (1.0 - std::exp(-static_cast<double>(fb[i]) * exposure));

                    double avg = (r + g + b) / 3.0;
                    r = avg + (r - avg) * saturationBoost;
                    g = avg + (g - avg) * saturationBoost;
                    b = avg + (b - avg) * saturationBoost;

                    win.SetPix(x, y, Hal::Rgb(r, g, b));
                }
            }
            win.TickPause(headless ? 0 : 1);
        }

        if (headless)
        {
            win.StopRecording();
            win.ToPng("gpu_tempest.png");
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_tempest.mp4 and gpu_tempest.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuTempest failed: ") + e.what());
        return 1;
    }
}
