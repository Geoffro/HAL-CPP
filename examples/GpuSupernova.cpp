// Showcase: a planet-killer explosion, staged like the big screen ones — a
// blinding core flash, a spherical shock wave, the signature flat
// equatorial ring racing out ahead of it (tilted at random each blast),
// and the object itself fracturing into ~200,000 star-like debris sparks
// that tear outward from the origin in clustered fiery plumes, billow in
// the fireball's turbulence, and cool from white-hot through orange to
// dying embers. Hot sparks streak flame licks behind them. The smoke and
// fire are a real fluid: an incompressible stable-fluids solver
// (FluidKernel3D — Stam's semi-Lagrangian scheme with a pressure
// projection, the standard graphics smoke model) runs on the GPU with
// density/temperature injected at the plume mouths and an analytic
// divergence source as the radial blast wind, and the resulting field is
// drawn as thousands of soft additive glow puffs (SphereGlow) that
// accumulate into a genuine luminous fog. The blast re-detonates on a
// cycle, and every detonation is different: unseeded RNG draws the
// fracture plumes, debris speeds, cooling rates, and the ring plane
// fresh each time.
//
// GPU usage is triple-barreled: every frame the debris turbulence is
// advected through a chaotic 3D ABC flow (FlowFieldKernel3D — the host
// adds only the cheap radial ballistics on top of the returned churn),
// the Navier-Stokes smoke evolves on a 64^3 grid (FluidKernel3D: advect,
// buoyancy, divergence + source, Jacobi pressure solve, project — all
// device-resident per batch), and the hardware Vulkan renderer
// (Scene3DRenderContext) depth-tests the ~200k-instance opaque sphere
// cloud then accumulates the additive glow pass over it. The camera
// shakes for the first moments after each detonation and slowly orbits
// the aftermath.
//
// It needs a live window (the Vulkan surface), but pass a frame count to
// record a clip via the renderer's opt-in frame capture (writes
// gpu_supernova.mp4 + a final .png).
//
// Usage:
//   GpuSupernova                   live window (re-detonates forever)
//   GpuSupernova <frames>          run <frames> frames, recording mp4+png
//   GpuSupernova <count> <frames>  override the debris count too
//
// One of the GPU showcases (see also GpuFlowField3D, Gpu3DReactionDiffusion,
// GpuPlasmaVortex, GpuTempest).

#include "Hal/Color.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/FlowFieldKernel3D.h"
#include "Hal/Gpu/FluidKernel3D.h"
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

namespace GpuSupernovaInternal
{
    struct Vec3d
    {
        double x;
        double y;
        double z;
    };

    // Blackbody-ish ember ramp for the debris, heat 1 = white-hot core,
    // heat 0 = burnt out. Deliberately over-bright at the top so fresh
    // debris reads as staring into the flash.
    int32_t EmberColor(double heat)
    {
        if (heat < 0.0)
        {
            heat = 0.0;
        }
        if (heat > 1.0)
        {
            heat = 1.0;
        }
        const double anchors[5][3] = {
            {0.03, 0.01, 0.00},
            {0.45, 0.05, 0.01},
            {1.00, 0.30, 0.02},
            {1.00, 0.75, 0.20},
            {1.00, 1.00, 0.95},
        };
        const int32_t stops = 5;
        double scaled = heat * (stops - 1);
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

    // Uniform direction on the unit sphere (rejection sample the unit
    // ball, then normalize).
    Vec3d RandomUnitVector(Hal::Rand &rand)
    {
        for (;;)
        {
            Vec3d v = {};
            v.x = rand.Double() * 2.0 - 1.0;
            v.y = rand.Double() * 2.0 - 1.0;
            v.z = rand.Double() * 2.0 - 1.0;
            double lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
            if ((lenSq > 1.0e-6) && (lenSq <= 1.0))
            {
                double invLen = 1.0 / std::sqrt(lenSq);
                v.x *= invLen;
                v.y *= invLen;
                v.z *= invLen;
                return v;
            }
        }
    }

    // Shortest signed displacement between two angles on the kernel's
    // 2*pi-periodic domain, so a particle crossing the wrap seam reads as
    // a small step rather than a full-domain teleport.
    double WrapDelta(double now,
                     double before)
    {
        const double twoPi = 6.28318530717958647692;
        double d = now - before;
        if (d > (twoPi * 0.5))
        {
            d -= twoPi;
        }
        if (d < (-twoPi * 0.5))
        {
            d += twoPi;
        }
        return d;
    }
}

using namespace GpuSupernovaInternal;

int main(int argc, char **argv)
{
    try
    {
        int32_t debrisCount = 200000;
        int32_t maxFrames = 0;
        // Same convention as GpuFlowField3D: one positive int is the frame
        // count; two are taken as debris count then frame count.
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
            debrisCount = numbers[0];
            maxFrames = numbers[1];
        }

        const double twoPi = 6.28318530717958647692;
        const double box = 40.0;
        const double cx = box * 0.5;
        const double cy = box * 0.5;
        const double cz = box * 0.5;

        // One full blast: detonation, expansion, ember fade, darkness,
        // then the next detonation.
        const int32_t cycleFrames = 430;

        // Debris ballistics. Launch speeds are drawn per particle at
        // detonation (biased slow, so a fast bright vanguard leads a
        // dense glowing core); drag brings everything to rest inside a
        // couple hundred frames.
        const double minLaunchSpeed = box * 0.0022;
        const double extraLaunchSpeed = box * 0.0085;
        const double drag = 0.985;

        // GPU ABC turbulence (FlowFieldKernel3D). The kernel advects each
        // spark's phase point through the flow; the wrap-safe delta,
        // scaled by a gain that decays as the fireball ages, is the churn
        // added on top of the radial flight. The phase seeds are spread
        // over a wide ball (kernelSeedRadius) so nearby debris ride
        // DIFFERENT streamlines — that decorrelation is what makes the
        // fireball billow into rolling lobes instead of drifting as one
        // rigid cloud.
        const float  coeffA = 1.7f;
        const float  coeffB = 1.3f;
        const float  coeffC = 1.0f;
        const float  turbDt = 0.016f;
        const int32_t turbSubsteps = 2;
        const double turbGain = box * 0.09;

        // Fracture plumes: debris bursts in clustered radial fingers
        // (drawn fresh each detonation), with a fraction left isotropic
        // as the haze between them. Per-plume speed/cooling scaling makes
        // whole lobes run hotter or cooler — the mottled, fiery look.
        const int32_t plumeCount = 70;
        const double  isotropicFraction = 0.30;

        // Fluid smoke/fire (FluidKernel3D — Stam's stable fluids, the
        // standard graphics smoke model). Density and temperature are
        // injected at the fracture plume mouths while the blast is young;
        // an analytic divergence source at the core is the radial blast
        // wind; buoyancy and the pressure projection do the billowing.
        // The smoke is drawn as the density field's iso-shell in tiny
        // voxel-sized spheres (a connected rolling surface, the same
        // trick Gpu3DReactionDiffusion uses), colored by temperature from
        // fire down to soot.
        const int32_t fluidDim = 64;
        const int32_t fluidSteps = 2;
        const int32_t injectFrames = 100;
        const float   fluidDt = 0.9f;
        const float   velocityDissipation = 0.992f;
        const float   densityDissipation = 0.996f;
        const float   temperatureDissipation = 0.985f;
        // Near-zero buoyancy: this is a space explosion — the smoke
        // spreads radially with the blast wind instead of rising. (Even
        // a small value here compounds against the very hot core and
        // launches the whole cloud upward.)
        const float   fluidBuoyancy = 0.001f;
        const int32_t pressureIterations = 16;
        const double  smokeThreshold = 0.08;

        // Flame licks: the hottest sampled sparks draw a short unlit
        // streak backward along their flight — directional tongues of
        // fire that die out as the debris cools past lickMinHeat.
        const int32_t lickStride = 40;
        const double  lickMinHeat = 0.5;
        const double  lickLength = 5.0;

        // Debris cooling / sparkle. Cooling rates are per particle (drawn
        // at detonation); twinkleChance is per particle per frame — the
        // random white flares that keep the cloud glittering.
        const double minCoolRate = 0.006;
        const double extraCoolRate = 0.010;
        const double twinkleChance = 0.004;

        // Shock wave, Praxis ring, and core flash timing/scale. Both
        // fronts fade on a slow envelope and their colors scale on a
        // sub-linear power of it, so they stay a searing blue-white for
        // most of their run instead of graying out immediately. Their
        // RADII are not independent speeds: every subsystem's front is
        // derived from the debris' own drag-decelerated ballistic curve
        // (see the shared front computation in the frame loop), so the
        // shell rides the fastest sparks, the ring runs proportionally
        // ahead, and the whole explosion decelerates as one object.
        const int32_t shellPoints = 6500;
        const double  shellFade = 0.012;
        const int32_t ringPoints = 1400;
        const double  ringFade = 0.012;
        const double  flashRadius0 = box * 0.16;
        const double  flashDecay = 0.11;
        const int32_t chunkCount = 160;
        const int32_t starCount = 360;

        Hal::Gpu::ComputeContext context;
        Hal::Gpu::FlowFieldKernel3D turbKernel(context, debrisCount);
        Hal::Gpu::FluidKernel3D fluid(context, fluidDim, fluidDim, fluidDim);
        Hal::Rand rand;

        // Debris state. World positions are integrated on the host;
        // kernelPos is the particle's phase point in the ABC flow's
        // periodic domain (xyz position + speed in w, the kernel's
        // layout).
        std::vector<float>  kernelPos(static_cast<size_t>(debrisCount) * 4, 0.0f);
        std::vector<float>  kernelPrev(static_cast<size_t>(debrisCount) * 4, 0.0f);
        std::vector<double> worldX(static_cast<size_t>(debrisCount), 0.0);
        std::vector<double> worldY(static_cast<size_t>(debrisCount), 0.0);
        std::vector<double> worldZ(static_cast<size_t>(debrisCount), 0.0);
        std::vector<Vec3d>  flightDir(static_cast<size_t>(debrisCount));
        std::vector<double> flightSpeed(static_cast<size_t>(debrisCount), 0.0);
        std::vector<double> coolRate(static_cast<size_t>(debrisCount), 0.0);
        std::vector<double> debrisRadius(static_cast<size_t>(debrisCount), 0.0);

        // Fluid scalar fields, interleaved (density, temperature) per
        // voxel; the velocity field lives only on the GPU.
        std::vector<float> fluidScalars(static_cast<size_t>(fluidDim * fluidDim * fluidDim) * 2, 0.0f);
        const double voxel = box / fluidDim;

        // Shock shell directions, fracture plumes, ring basis, and
        // per-blast state, redrawn at every detonation.
        std::vector<Vec3d>  shellDir(static_cast<size_t>(shellPoints));
        std::vector<Vec3d>  plumeDir(static_cast<size_t>(plumeCount));
        std::vector<double> plumeSpread(static_cast<size_t>(plumeCount), 0.0);
        std::vector<double> plumeSpeedScale(static_cast<size_t>(plumeCount), 1.0);
        std::vector<double> plumeCoolScale(static_cast<size_t>(plumeCount), 1.0);
        Vec3d ringU = {};
        Vec3d ringV = {};

        // A fixed starfield backdrop, dim and slightly blue, so the blast
        // has something to blind.
        std::vector<Vec3d>   starPos(static_cast<size_t>(starCount));
        std::vector<double>  starSize(static_cast<size_t>(starCount), 0.0);
        std::vector<int32_t> starColor(static_cast<size_t>(starCount), 0);
        for (int32_t i = 0; i < starCount; i++)
        {
            size_t si = static_cast<size_t>(i);
            starPos[si].x = rand.Double(box);
            starPos[si].y = rand.Double(box);
            starPos[si].z = rand.Double(box);
            starSize[si] = box * (0.0012 + 0.0018 * rand.Double());
            double glow = 0.30 + 0.45 * rand.Double();
            starColor[si] = Hal::Rgb(glow * 0.9, glow * 0.95, glow);
        }

        Hal::Gpu::Window               window(960, 720, "GPU Supernova (fracturing explosion)");
        Hal::Gpu::Scene3DRenderContext scene(window, box, box, box);
        scene.SetCamera(0.6, 0.25, box * 1.55);

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
        int32_t age = cycleFrames;
        while (!window.ShouldClose() && (!capture || (frame < maxFrames)))
        {
            window.PollEvents();

            // Detonate: rebuild every per-particle draw from fresh
            // entropy, so no two blasts fracture the same way.
            if (age >= cycleFrames)
            {
                age = 0;
                const double kernelCenter = twoPi * 0.5;
                const double kernelSeedRadius = twoPi * 0.22;

                for (int32_t p = 0; p < plumeCount; p++)
                {
                    size_t pi = static_cast<size_t>(p);
                    plumeDir[pi] = RandomUnitVector(rand);
                    plumeSpread[pi] = 0.12 + 0.45 * rand.Double();
                    plumeSpeedScale[pi] = 0.65 + 0.75 * rand.Double();
                    plumeCoolScale[pi] = 0.70 + 0.60 * rand.Double();
                }

                for (int32_t i = 0; i < debrisCount; i++)
                {
                    size_t si = static_cast<size_t>(i);
                    worldX[si] = cx;
                    worldY[si] = cy;
                    worldZ[si] = cz;

                    double bias = rand.Double();
                    flightSpeed[si] = minLaunchSpeed + extraLaunchSpeed * bias * bias;
                    coolRate[si] = minCoolRate + extraCoolRate * rand.Double();

                    // Most debris joins a fracture plume — a clustered
                    // radial finger sharing a direction cone and a
                    // hotter-or-cooler temperament — while the isotropic
                    // remainder fills the fireball body between fingers.
                    if (rand.Double() < isotropicFraction)
                    {
                        flightDir[si] = RandomUnitVector(rand);
                    }
                    else
                    {
                        size_t pi = static_cast<size_t>(rand.Int(plumeCount));
                        Vec3d jitter = RandomUnitVector(rand);
                        double spread = plumeSpread[pi] * rand.Double();
                        Vec3d dir = {};
                        dir.x = plumeDir[pi].x + jitter.x * spread;
                        dir.y = plumeDir[pi].y + jitter.y * spread;
                        dir.z = plumeDir[pi].z + jitter.z * spread;
                        double len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
                        dir.x /= len;
                        dir.y /= len;
                        dir.z /= len;
                        flightDir[si] = dir;
                        flightSpeed[si] *= plumeSpeedScale[pi];
                        coolRate[si] *= plumeCoolScale[pi];
                    }

                    // The first chunkCount particles are the big slow
                    // fragments of the object itself; the rest are
                    // star-like sparks.
                    if (i < chunkCount)
                    {
                        debrisRadius[si] = box * (0.004 + 0.008 * rand.Double());
                        flightSpeed[si] *= 0.45;
                        coolRate[si] *= 0.55;
                    }
                    else
                    {
                        debrisRadius[si] = box * (0.0014 + 0.0008 * rand.Double());
                    }
                    Vec3d seed = RandomUnitVector(rand);
                    double seedR = kernelSeedRadius * rand.Double();
                    kernelPos[static_cast<size_t>(4 * i)]     = static_cast<float>(kernelCenter + seed.x * seedR);
                    kernelPos[static_cast<size_t>(4 * i + 1)] = static_cast<float>(kernelCenter + seed.y * seedR);
                    kernelPos[static_cast<size_t>(4 * i + 2)] = static_cast<float>(kernelCenter + seed.z * seedR);
                    kernelPos[static_cast<size_t>(4 * i + 3)] = 0.0f;
                }

                // The old blast's smoke clears with the new detonation
                // (the GPU-resident velocity field is left as-is — a
                // little residual churn only helps the next fireball).
                std::fill(fluidScalars.begin(), fluidScalars.end(), 0.0f);

                for (int32_t i = 0; i < shellPoints; i++)
                {
                    shellDir[static_cast<size_t>(i)] = RandomUnitVector(rand);
                }

                // Random ring plane: a random normal, completed into an
                // orthonormal basis for the ring circle.
                Vec3d normal = RandomUnitVector(rand);
                Vec3d helper = {};
                if (std::abs(normal.x) < 0.9)
                {
                    helper.x = 1.0;
                }
                else
                {
                    helper.y = 1.0;
                }
                ringU.x = normal.y * helper.z - normal.z * helper.y;
                ringU.y = normal.z * helper.x - normal.x * helper.z;
                ringU.z = normal.x * helper.y - normal.y * helper.x;
                double uLen = std::sqrt(ringU.x * ringU.x + ringU.y * ringU.y + ringU.z * ringU.z);
                ringU.x /= uLen;
                ringU.y /= uLen;
                ringU.z /= uLen;
                ringV.x = normal.y * ringU.z - normal.z * ringU.y;
                ringV.y = normal.z * ringU.x - normal.x * ringU.z;
                ringV.z = normal.x * ringU.y - normal.y * ringU.x;
            }

            // GPU: churn every spark's phase point through the ABC flow.
            kernelPrev = kernelPos;
            turbKernel.RunSteps(kernelPos, turbSubsteps, coeffA, coeffB, coeffC, turbDt);

            // Host: radial ballistics + the GPU churn, and cooling. The
            // turbulence gain decays as the fireball ages so late embers
            // drift instead of boiling.
            double turbEnvelope = turbGain * (0.15 + 0.85 * std::exp(-age / 140.0));
            for (int32_t i = 0; i < debrisCount; i++)
            {
                size_t si = static_cast<size_t>(i);
                size_t ki = static_cast<size_t>(4 * i);
                worldX[si] += flightDir[si].x * flightSpeed[si] + WrapDelta(kernelPos[ki], kernelPrev[ki]) * turbEnvelope;
                worldY[si] += flightDir[si].y * flightSpeed[si] + WrapDelta(kernelPos[ki + 1], kernelPrev[ki + 1]) * turbEnvelope;
                worldZ[si] += flightDir[si].z * flightSpeed[si] + WrapDelta(kernelPos[ki + 2], kernelPrev[ki + 2]) * turbEnvelope;
                flightSpeed[si] *= drag;
            }

            // One shared ballistic front: how far a spark launched at
            // speed v0 has traveled after `age` frames of drag. EVERY
            // subsystem's radius rides this same curve — the bulk
            // fireball at the mean spark speed, the shock shell at the
            // fastest vanguard's, the ring proportionally ahead, the
            // smoke injection at the bulk — so the whole explosion
            // expands from one origin and decelerates as one object
            // instead of three independent spreads.
            double dragSum = (1.0 - std::pow(drag, age + 1.0)) / (1.0 - drag);
            double bulkFront = (minLaunchSpeed + extraLaunchSpeed * 0.35) * dragSum;
            double shellRadius = (minLaunchSpeed + extraLaunchSpeed * 0.90) * dragSum;
            double ringRadius = shellRadius * 1.35;

            // Fluid: while the blast is young, inject hot dense smoke at
            // the core and at each plume mouth (stamped as small blobs in
            // voxel space, riding out with the shared front), then run
            // the stable-fluids solver — the divergence source is the
            // radial blast wind, decaying as the explosion ages.
            if (age < injectFrames)
            {
                double linearFade = 1.0 - static_cast<double>(age) / injectFrames;
                double injectFade = linearFade * linearFade;
                double plumeFront = 2.0 + (bulkFront * 0.95) / voxel;
                if (plumeFront > (fluidDim * 0.38))
                {
                    plumeFront = fluidDim * 0.38;
                }
                for (int32_t p = -1; p < plumeCount; p++)
                {
                    double bx = fluidDim * 0.5;
                    double by = fluidDim * 0.5;
                    double bz = fluidDim * 0.5;
                    double strength = 1.0;
                    if (p >= 0)
                    {
                        // Only every other plume feeds smoke — all 70
                        // would fill a closed shell and hide the sparks.
                        if ((p % 2) != 0)
                        {
                            continue;
                        }
                        size_t pi = static_cast<size_t>(p);
                        bx += plumeDir[pi].x * plumeFront;
                        by += plumeDir[pi].y * plumeFront;
                        bz += plumeDir[pi].z * plumeFront;
                        strength = 0.5 * plumeSpeedScale[pi];
                    }
                    int32_t ix = static_cast<int32_t>(bx);
                    int32_t iy = static_cast<int32_t>(by);
                    int32_t iz = static_cast<int32_t>(bz);
                    for (int32_t dx = -2; dx <= 2; dx++)
                    {
                        for (int32_t dy = -2; dy <= 2; dy++)
                        {
                            for (int32_t dz = -2; dz <= 2; dz++)
                            {
                                int32_t x = ix + dx;
                                int32_t y = iy + dy;
                                int32_t z = iz + dz;
                                if ((x < 1) || (x >= (fluidDim - 1)) || (y < 1) || (y >= (fluidDim - 1)) || (z < 1) || (z >= (fluidDim - 1)))
                                {
                                    continue;
                                }
                                double falloff = std::exp(-static_cast<double>(dx * dx + dy * dy + dz * dz) * 0.35);
                                size_t cell = static_cast<size_t>(((x * fluidDim + y) * fluidDim + z) * 2);
                                fluidScalars[cell] += static_cast<float>(strength * injectFade * falloff);
                                fluidScalars[cell + 1] += static_cast<float>(strength * injectFade * falloff);
                            }
                        }
                    }
                }
            }

            Hal::Gpu::FluidParams fluidParams = {};
            fluidParams.dt = fluidDt;
            fluidParams.velocityDissipation = velocityDissipation;
            fluidParams.densityDissipation = densityDissipation;
            fluidParams.temperatureDissipation = temperatureDissipation;
            fluidParams.buoyancy = fluidBuoyancy;
            fluidParams.ambientTemperature = 0.0f;
            fluidParams.sourceX = fluidDim * 0.5f;
            fluidParams.sourceY = fluidDim * 0.5f;
            fluidParams.sourceZ = fluidDim * 0.5f;
            fluidParams.sourceRadius = fluidDim * 0.08f;
            fluidParams.sourceStrength = static_cast<float>(0.9 * std::exp(-age * 0.06));
            fluidParams.pressureIterations = pressureIterations;
            fluid.RunSteps(fluidScalars, fluidSteps, fluidParams);

            // Camera: slow orbit, plus a hard shake in the first moments
            // after the detonation.
            yaw += 0.004;
            double shakeYaw = 0.0;
            double shakePitch = 0.0;
            if (age < 24)
            {
                double amp = 0.022 * (1.0 - age / 24.0);
                shakeYaw = (rand.Double() * 2.0 - 1.0) * amp;
                shakePitch = (rand.Double() * 2.0 - 1.0) * amp;
            }
            scene.SetCamera(yaw + shakeYaw, 0.25 + 0.10 * std::sin(frame * 0.005) + shakePitch, box * 1.55);
            scene.Clear(Hal::Rgb(0.005, 0.005, 0.015));

            for (int32_t i = 0; i < starCount; i++)
            {
                size_t si = static_cast<size_t>(i);
                scene.Sphere(starPos[si].x, starPos[si].y, starPos[si].z, starSize[si], starColor[si]);
            }

            // Debris: heat holds white-hot through the flash, then decays
            // per particle; a random few sparks (never the big chunks —
            // a chunk flaring to 1.8x reads as a glitch, not a glint)
            // flare white each frame.
            double coolAge = (age > 12) ? static_cast<double>(age - 12) : 0.0;
            for (int32_t i = 0; i < debrisCount; i++)
            {
                size_t si = static_cast<size_t>(i);
                double heat = std::exp(-coolAge * coolRate[si]);
                double radius = debrisRadius[si];
                int32_t color;
                if ((i >= chunkCount) && (rand.Double() < twinkleChance))
                {
                    color = Hal::Rgb(1.0, 1.0, 1.0);
                    radius *= 1.8;
                }
                else
                {
                    color = EmberColor(heat);
                }
                scene.Sphere(worldX[si], worldY[si], worldZ[si], radius, color);

                // Flame licks: while a sampled spark is still hot, streak
                // an unlit line back along its flight — a tongue of fire
                // that shortens and dies as the spark cools and slows.
                if (((i % lickStride) == 0) && (heat > lickMinHeat))
                {
                    double lick = flightSpeed[si] * lickLength;
                    scene.Line(worldX[si], worldY[si], worldZ[si],
                               worldX[si] - flightDir[si].x * lick,
                               worldY[si] - flightDir[si].y * lick,
                               worldZ[si] - flightDir[si].z * lick,
                               EmberColor(heat));
                }
            }

            // Smoke/fire: every fluid voxel carrying density becomes a
            // soft additive glow puff (SphereGlow — no hard silhouette,
            // depth-tested against the solids but accumulating with
            // itself), overlapping its neighbors so the field reads as a
            // continuous luminous fog rather than discrete balls. Color
            // rides the local temperature from white-hot down through
            // ember red; intensity rides the density and stays small
            // because thousands of puffs overlap.
            for (int32_t x = 1; x < (fluidDim - 1); x++)
            {
                for (int32_t y = 1; y < (fluidDim - 1); y++)
                {
                    for (int32_t z = 1; z < (fluidDim - 1); z++)
                    {
                        size_t cell = static_cast<size_t>(((x * fluidDim + y) * fluidDim + z) * 2);
                        double density = fluidScalars[cell];
                        if (density < smokeThreshold)
                        {
                            continue;
                        }
                        double temp = fluidScalars[cell + 1];
                        double heat = temp / (temp + 0.8);
                        double intensity = 0.05 * (density / (density + 1.2)) * (0.25 + 0.75 * heat);
                        scene.SphereGlow((x + 0.5) * voxel,
                                         (y + 0.5) * voxel,
                                         (z + 0.5) * voxel,
                                         voxel * 1.7,
                                         EmberColor(heat),
                                         intensity);
                    }
                }
            }

            // Shock shell: an expanding sphere of soft blue-white glow
            // points riding the shared ballistic front just ahead of the
            // fastest debris — additive, so where the shell passes
            // through the fireball the two brighten each other instead
            // of one occluding the other. Color rides pow(glow, 0.4) so
            // the front stays near white for most of its run; drawing
            // stops once it has flown well past the domain.
            double shellGlow = std::exp(-age * shellFade);
            if ((shellGlow > 0.05) && (shellRadius < (box * 0.85)))
            {
                double shellBright = std::pow(shellGlow, 0.4);
                int32_t shellColor = Hal::Rgb(0.80, 0.92, 1.0);
                for (int32_t i = 0; i < shellPoints; i++)
                {
                    size_t si = static_cast<size_t>(i);
                    scene.SphereGlow(cx + shellDir[si].x * shellRadius,
                                     cy + shellDir[si].y * shellRadius,
                                     cz + shellDir[si].z * shellRadius,
                                     box * 0.007,
                                     shellColor,
                                     0.38 * shellBright);
                }
            }

            // Praxis ring: the flat blast ring running proportionally
            // ahead of the shell on the same ballistic curve, in its own
            // randomly tilted plane — a searing blade of additive light,
            // the brightest thing on screen after the flash itself.
            double ringGlow = std::exp(-age * ringFade);
            if ((ringGlow > 0.05) && (ringRadius < (box * 0.85)))
            {
                double ringBright = std::pow(ringGlow, 0.35);
                int32_t ringColor = Hal::Rgb(0.90, 1.0, 1.0);
                for (int32_t i = 0; i < ringPoints; i++)
                {
                    double theta = twoPi * i / ringPoints;
                    double cu = std::cos(theta) * ringRadius;
                    double cv = std::sin(theta) * ringRadius;
                    scene.SphereGlow(cx + ringU.x * cu + ringV.x * cv,
                                     cy + ringU.y * cu + ringV.y * cv,
                                     cz + ringU.z * cu + ringV.z * cv,
                                     box * 0.012,
                                     ringColor,
                                     0.8 * ringBright);
                }
            }

            // Core flash: a blinding glow collapsing over the first
            // moments (nested puffs give it a hot core inside a wide
            // halo), with star-filter spikes through the center.
            double flashRadius = flashRadius0 * std::exp(-age * flashDecay);
            if (flashRadius > (box * 0.004))
            {
                scene.SphereGlow(cx, cy, cz, flashRadius * 0.6, Hal::Rgb(1.0, 1.0, 1.0), 3.0);
                scene.SphereGlow(cx, cy, cz, flashRadius * 1.4, Hal::Rgb(1.0, 0.9, 0.7), 1.2);
                scene.SphereGlow(cx, cy, cz, flashRadius * 2.6, Hal::Rgb(1.0, 0.6, 0.3), 0.4);
                double spike = flashRadius * 4.0;
                double spikeGlow = flashRadius / flashRadius0;
                int32_t spikeColor = Hal::Rgb(spikeGlow, spikeGlow, 0.9 * spikeGlow);
                scene.Line(cx - spike, cy, cz, cx + spike, cy, cz, spikeColor);
                scene.Line(cx, cy - spike, cz, cx, cy + spike, cz, spikeColor);
                scene.Line(cx, cy, cz - spike, cx, cy, cz + spike, spikeColor);
            }

            bool presented = scene.Render();
            if (capture && presented)
            {
                if (scene.ReadCapturedFrame(lastFrame, lastWidth, lastHeight))
                {
                    if (!recorder)
                    {
                        recorder = std::make_unique<Hal::Examples::Mp4Recorder>("gpu_supernova.mp4", lastWidth, lastHeight, 30, 1);
                    }
                    recorder->AddFrame(lastFrame);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
            frame++;
            age++;
        }

        if (capture && !lastFrame.empty())
        {
            recorder.reset();
            Hal::ExportPng("gpu_supernova.png", lastWidth, lastHeight, lastFrame);
            Hal::Logger::Instance().Log(Hal::LogSeverity::Info, "Wrote gpu_supernova.mp4 and gpu_supernova.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GpuSupernova failed: ") + e.what());
        return 1;
    }
}
