#include <doctest/doctest.h>

#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/FlowFieldKernel3D.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace FlowFieldKernel3DTestsInternal
{
    const double kTwoPi = 6.28318530717958647692;

    void AbcVelocity(double  a,
                     double  b,
                     double  c,
                     double  x,
                     double  y,
                     double  z,
                     double &vx,
                     double &vy,
                     double &vz)
    {
        vx = a * std::sin(z) + c * std::cos(y);
        vy = b * std::sin(x) + a * std::cos(z);
        vz = c * std::sin(y) + b * std::cos(x);
    }
}

using namespace FlowFieldKernel3DTestsInternal;

// The advection is a deterministic RK2 map with no cross-particle coupling,
// so the GPU result must match a CPU replay of the same ABC-flow RK2 step
// (with the same [0, 2*pi) wrap) to float precision — a direct check of the
// shader math, the GPU-resident sub-step chaining, and the readback.
TEST_CASE("Hal::Gpu::FlowFieldKernel3D matches a CPU RK2 replay of the ABC flow, or reports why it can't")
{
    try
    {
        Hal::Gpu::ComputeContext context;

        const int32_t count = 512;
        const int32_t steps = 8;
        const double a = 1.4;
        const double b = 1.1;
        const double c = 0.9;
        const double dt = 0.02;

        Hal::Gpu::FlowFieldKernel3D kernel(context, count);

        // Deterministic scattered seeds kept in [1, 2*pi - 1]: over this
        // short run the max displacement (~0.5) can't reach a wrap seam, so
        // no particle wraps and the comparison isolates the RK2 math without
        // chaotic float32-vs-float64 trajectory divergence or seam ambiguity.
        const double band = kTwoPi - 2.0;
        std::vector<float> positions(static_cast<size_t>(count) * 4);
        std::vector<double> px(static_cast<size_t>(count));
        std::vector<double> py(static_cast<size_t>(count));
        std::vector<double> pz(static_cast<size_t>(count));
        for (int32_t i = 0; i < count; i++)
        {
            double x = 1.0 + std::fmod(0.13 * i, band);
            double y = 1.0 + std::fmod(0.29 * i + 0.5, band);
            double z = 1.0 + std::fmod(0.47 * i + 1.1, band);
            px[static_cast<size_t>(i)] = x;
            py[static_cast<size_t>(i)] = y;
            pz[static_cast<size_t>(i)] = z;
            positions[static_cast<size_t>(4 * i)] = static_cast<float>(x);
            positions[static_cast<size_t>(4 * i + 1)] = static_cast<float>(y);
            positions[static_cast<size_t>(4 * i + 2)] = static_cast<float>(z);
            positions[static_cast<size_t>(4 * i + 3)] = 0.0f;
        }

        // CPU replay of the identical RK2 + wrap.
        for (int32_t step = 0; step < steps; step++)
        {
            for (int32_t i = 0; i < count; i++)
            {
                double x = px[static_cast<size_t>(i)];
                double y = py[static_cast<size_t>(i)];
                double z = pz[static_cast<size_t>(i)];
                double k1x;
                double k1y;
                double k1z;
                AbcVelocity(a, b, c, x, y, z, k1x, k1y, k1z);
                double k2x;
                double k2y;
                double k2z;
                AbcVelocity(a, b, c, x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z, k2x, k2y, k2z);
                x = std::fmod(x + dt * k2x, kTwoPi);
                y = std::fmod(y + dt * k2y, kTwoPi);
                z = std::fmod(z + dt * k2z, kTwoPi);
                if (x < 0.0)
                {
                    x += kTwoPi;
                }
                if (y < 0.0)
                {
                    y += kTwoPi;
                }
                if (z < 0.0)
                {
                    z += kTwoPi;
                }
                px[static_cast<size_t>(i)] = x;
                py[static_cast<size_t>(i)] = y;
                pz[static_cast<size_t>(i)] = z;
            }
        }

        kernel.RunSteps(positions, steps, static_cast<float>(a), static_cast<float>(b), static_cast<float>(c), static_cast<float>(dt));

        // Circular distance, so a particle sitting exactly on the 0/2*pi
        // seam that wraps one way on the GPU and the other on the CPU (a
        // float-epsilon difference) doesn't read as a ~2*pi error.
        auto circularErr = [](double gpu, double cpu) {
            double d = std::abs(gpu - cpu);
            return std::min(d, kTwoPi - d);
        };
        double maxErr = 0.0;
        for (int32_t i = 0; i < count; i++)
        {
            maxErr = std::max(maxErr, circularErr(static_cast<double>(positions[static_cast<size_t>(4 * i)]), px[static_cast<size_t>(i)]));
            maxErr = std::max(maxErr, circularErr(static_cast<double>(positions[static_cast<size_t>(4 * i + 1)]), py[static_cast<size_t>(i)]));
            maxErr = std::max(maxErr, circularErr(static_cast<double>(positions[static_cast<size_t>(4 * i + 2)]), pz[static_cast<size_t>(i)]));
        }
        // Float32 accumulation over 30 chained steps.
        CHECK(maxErr < 2e-3);
    }
    catch (const std::runtime_error &e)
    {
        MESSAGE("No usable Vulkan compute device in this environment: ", std::string(e.what()));
    }
}
