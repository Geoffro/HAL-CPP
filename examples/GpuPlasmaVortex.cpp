// Showcase: an abstract electric-plasma storm — pure light and color, no
// biology, no organism. 200,000 charged sparks are torn between four
// counter-spinning vortices anchored to the four corners of the canvas,
// each with its own neon signature (gold, violet, magenta, cyan); every
// spark bleeds whichever vortex it's currently nearest into three
// GPU-resident trail fields (one per RGB channel), so each corner blooms
// its own bold color and the seams where two vortices' filaments cross
// add into blinding white-hot light. Each vortex core stays a dark "eye"
// — sparks orbit it but never fill it, like a hurricane center.
//
// This is the "hybrid" model pushed as far as it goes in 2D: BOTH halves
// stay GPU-resident every frame. AgentForceKernel2D gives the spark cloud
// real texture via bucketed neighbor repulsion (as in GpuSwarm), while
// DiffusionKernel2D blooms and fades all three color channels
// (RunSteps, multiple sub-steps per frame, as in GpuSlimeMold — but three
// independent fields instead of one). Neither half is decorative: the
// repulsion keeps the sparks from collapsing into the vortex cores, and
// the three-channel bloom is what turns point samples into luminous,
// overlapping color washes.
//
// Usage:
//   GpuPlasmaVortex             live window
//   GpuPlasmaVortex headless    no window; writes gpu_plasma_vortex.mp4 + png
//   GpuPlasmaVortex <N>         override the spark count (also works headless)
//
// One of the GPU showcases (see also GpuSwarm, GpuSlimeMold, GpuInkFlow).

#include "Hal/Color.h"
#include "Hal/Gpu/AgentForceKernel2D.h"
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

namespace GpuPlasmaVortexInternal
{
    // One spinning storm cell. Sparks near it are swept along its
    // tangent (spin flips the direction, so neighboring vortices fight
    // each other) and drawn weakly toward its core; its tint is bled into
    // the trail fields by whichever sparks are currently closest to it
    // (see the per-agent tint blend in the main loop). Each vortex owns a
    // fixed quadrant (baseX, baseY) and only wobbles locally within it
    // (wobbleRadius) — keeping the four basins geographically separate is
    // what keeps their colors bold instead of averaging into a single
    // grayish wash everywhere.
    struct Vortex
    {
        double baseX;
        double baseY;
        double wobbleRadius;
        double angularSpeed;
        double phase;
        double spin;
        float  tintR;
        float  tintG;
        float  tintB;
    };

    // Bold, fully-saturated neon tints — deliberately not a natural
    // biological palette (no greens-as-life, no flesh tones).
    const float g_vortexTint[4][3] = {
        {1.00f, 0.05f, 0.65f},
        {0.05f, 1.00f, 0.85f},
        {1.00f, 0.85f, 0.05f},
        {0.55f, 0.05f, 1.00f},
    };

    // One quadrant anchor per vortex (as a fraction of nx/ny), so the
    // four storm cells sit in the four corners of the canvas instead of
    // sharing a ring around one common center.
    const double g_vortexAnchor[4][2] = {
        {0.28, 0.28},
        {0.72, 0.28},
        {0.28, 0.72},
        {0.72, 0.72},
    };

    // Deliberately mismatched angular speeds so the four vortices' local
    // wobble never falls into a repeating, synchronized pattern.
    const double g_vortexAngularSpeed[4] = {0.011, 0.027, 0.017, 0.035};

    void ComputeVortexCenters(const std::vector<Vortex> &vortices,
                              double                      driftX,
                              double                      driftY,
                              int32_t                     frame,
                              std::vector<double>        &outX,
                              std::vector<double>        &outY)
    {
        for (size_t v = 0; v < vortices.size(); v++)
        {
            double angle = vortices[v].phase + vortices[v].angularSpeed * frame;
            outX[v] = vortices[v].baseX + driftX + vortices[v].wobbleRadius * std::cos(angle);
            outY[v] = vortices[v].baseY + driftY + vortices[v].wobbleRadius * std::sin(angle);
        }
    }
}

using namespace GpuPlasmaVortexInternal;

int main(int argc, char **argv)
{
    try
    {
        bool headless = false;
        int32_t requestedCount = 200000;
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
        const int32_t nx = 480;
        const int32_t ny = 480;
        const int32_t visScale = 2;
        const int32_t totalFrames = headless ? 360 : 100000;

        // Spark repulsion (GPU AgentForceKernel2D), same shape as GpuSwarm.
        const float  interactionRad = 1.5f;
        const float  springConstant = 1.7f;
        const double dt = 0.5;
        const double damping = 0.93;
        const double repelScale = 1.0;

        // Vortex swirl/pull. swirlStrength dominates so sparks orbit
        // rather than fall straight in; pullStrength keeps each basin
        // tight around its quadrant anchor. falloff is a fraction of the
        // quadrant spacing (~0.44*nx) so each vortex's pull is strongest
        // within its own corner and only blends gently at the seams
        // between quadrants, instead of all four summing across the
        // whole canvas into a flat gray wash.
        const double vortexWobbleRadius = nx * 0.07;
        const double spawnRadius = nx * 0.17;
        const double swirlStrength = 2.6;
        const double pullStrength = 0.55;
        const double falloff = 34.0;

        // Trail deposit / bloom. Kept modest per frame — decay pulls the
        // field back down every step, so brightness settles at a steady
        // state instead of climbing until every channel clips to white.
        const float  depositAmount = 1.1f;
        const float  diffRate = 0.22f;
        const int32_t diffusionSubSteps = 2;
        const double trailDecay = 0.85;
        const double exposure = 0.14;

        // Pushed away from the average toward each channel's extreme so
        // the four vortex tints read as bold, separated neon hues instead
        // of blending into a grayish average wherever fields overlap.
        const double saturationBoost = 1.6;

        Hal::Gpu::ComputeContext context;
        Hal::Gpu::AgentForceKernel2D forceKernel(context, nx, ny, numAgents);
        Hal::Gpu::DiffusionKernel2D diffusionKernel(context, nx, ny);
        Hal::Rand rand;

        std::vector<Vortex> vortices(4);
        for (size_t v = 0; v < vortices.size(); v++)
        {
            vortices[v].baseX        = nx * g_vortexAnchor[v][0];
            vortices[v].baseY        = ny * g_vortexAnchor[v][1];
            vortices[v].wobbleRadius = vortexWobbleRadius;
            vortices[v].angularSpeed = g_vortexAngularSpeed[v];
            vortices[v].phase        = 2.0 * M_PI * static_cast<double>(v) / static_cast<double>(vortices.size());
            vortices[v].spin         = ((v == 0) || (v == 3)) ? 1.0 : -1.0;
            vortices[v].tintR        = g_vortexTint[v][0];
            vortices[v].tintG        = g_vortexTint[v][1];
            vortices[v].tintB        = g_vortexTint[v][2];
        }

        std::vector<double> vortexX(vortices.size());
        std::vector<double> vortexY(vortices.size());
        ComputeVortexCenters(vortices, 0.0, 0.0, 0, vortexX, vortexY);

        std::vector<float> positions(static_cast<size_t>(numAgents) * 2);
        std::vector<float> velocities(static_cast<size_t>(numAgents) * 2, 0.0f);
        std::vector<float> radii(static_cast<size_t>(numAgents), 0.7f);
        std::vector<float> sparkTintR(static_cast<size_t>(numAgents));
        std::vector<float> sparkTintG(static_cast<size_t>(numAgents));
        std::vector<float> sparkTintB(static_cast<size_t>(numAgents));

        // Seed each spark near a random vortex's basin so the cloud
        // starts already spread across all four corners. sparkTint is
        // recomputed every frame from the spark's current position (see
        // the integration loop below), not fixed at spawn — that is what
        // keeps a quadrant's color matching whichever sparks are
        // currently in it even as they get handed between vortices.
        for (int32_t i = 0; i < numAgents; i++)
        {
            int32_t home = static_cast<int32_t>(rand.Double(static_cast<double>(vortices.size())));
            if (home >= static_cast<int32_t>(vortices.size()))
            {
                home = static_cast<int32_t>(vortices.size()) - 1;
            }
            double angle = rand.Double(2.0 * M_PI);
            double radius = rand.Double(spawnRadius);
            positions[static_cast<size_t>(2 * i)]     = static_cast<float>(vortexX[static_cast<size_t>(home)] + radius * std::cos(angle));
            positions[static_cast<size_t>(2 * i + 1)] = static_cast<float>(vortexY[static_cast<size_t>(home)] + radius * std::sin(angle));
        }

        std::vector<float> fr(static_cast<size_t>(nx * ny), 0.0f);
        std::vector<float> fg(static_cast<size_t>(nx * ny), 0.0f);
        std::vector<float> fb(static_cast<size_t>(nx * ny), 0.0f);

        Hal::Examples::GridWindow win("GPU Plasma Vortex (charged spark storm)", nx, ny, visScale, !headless);
        if (headless)
        {
            win.StartRecording("gpu_plasma_vortex.mp4", 30);
        }

        for (int32_t frame = 0; (frame < totalFrames) && !win.IsClosed(); frame++)
        {
            // GPU: neighbor repulsion so the storm has texture instead of
            // collapsing into four points.
            forceKernel.UploadAgents(positions, radii, numAgents);
            forceKernel.Dispatch(interactionRad, springConstant, true, true);
            std::vector<float> forces = forceKernel.DownloadVelocityDeltas();

            // The whole storm drifts slowly across the canvas so the
            // pattern never settles into a static shape, without ever
            // drifting far enough to break the four quadrants apart.
            double driftX = nx * 0.03 * std::cos(frame * 0.0016);
            double driftY = ny * 0.03 * std::sin(frame * 0.0021);
            ComputeVortexCenters(vortices, driftX, driftY, frame, vortexX, vortexY);

            // Host: integrate. Repulsion (GPU) + swirl/pull from every
            // vortex, nearest-weighted so basins stay distinct but never
            // fully independent — sparks get handed between cells.
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

                // tintWeight is the force weight squared — a sharper
                // falloff than the force itself, so the deposited color
                // stays crisp within a quadrant while the swirl/pull
                // motion stays smooth across it.
                double tintR = 0.0;
                double tintG = 0.0;
                double tintB = 0.0;
                double tintWeightSum = 0.0;
                for (size_t v = 0; v < vortices.size(); v++)
                {
                    double dx = vortexX[v] - px;
                    double dy = vortexY[v] - py;
                    double dist = std::sqrt(dx * dx + dy * dy) + 1.0;
                    double weight = falloff / (dist + falloff);
                    double ux = dx / dist;
                    double uy = dy / dist;
                    double tx = -uy * vortices[v].spin;
                    double ty = ux * vortices[v].spin;
                    vx += (ux * pullStrength + tx * swirlStrength) * weight;
                    vy += (uy * pullStrength + ty * swirlStrength) * weight;

                    double tintWeight = weight * weight;
                    tintR += vortices[v].tintR * tintWeight;
                    tintG += vortices[v].tintG * tintWeight;
                    tintB += vortices[v].tintB * tintWeight;
                    tintWeightSum += tintWeight;
                }
                sparkTintR[static_cast<size_t>(i)] = static_cast<float>(tintR / tintWeightSum);
                sparkTintG[static_cast<size_t>(i)] = static_cast<float>(tintG / tintWeightSum);
                sparkTintB[static_cast<size_t>(i)] = static_cast<float>(tintB / tintWeightSum);

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

            // Bleed each spark's tint into the three trail fields with a
            // soft 3x3 splat, so deposits read as glow, not single pixels.
            for (int32_t i = 0; i < numAgents; i++)
            {
                int32_t x = static_cast<int32_t>(positions[static_cast<size_t>(2 * i)]);
                int32_t y = static_cast<int32_t>(positions[static_cast<size_t>(2 * i + 1)]);
                float tr = sparkTintR[static_cast<size_t>(i)] * depositAmount;
                float tg = sparkTintG[static_cast<size_t>(i)] * depositAmount;
                float tb = sparkTintB[static_cast<size_t>(i)] * depositAmount;
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

            // GPU: bloom and fade all three channels, entirely
            // GPU-resident for the requested sub-steps.
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
                    double r = 1.0 - std::exp(-static_cast<double>(fr[i]) * exposure);
                    double g = 1.0 - std::exp(-static_cast<double>(fg[i]) * exposure);
                    double b = 1.0 - std::exp(-static_cast<double>(fb[i]) * exposure);

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
            win.ToPng("gpu_plasma_vortex.png");
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_plasma_vortex.mp4 and gpu_plasma_vortex.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuPlasmaVortex failed: ") + e.what());
        return 1;
    }
}
