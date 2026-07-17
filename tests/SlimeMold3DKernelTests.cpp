#include <doctest/doctest.h>

#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/SlimeMold3DKernel.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace SlimeMold3DKernelTestsInternal
{
    int32_t WrapIndex(int32_t v,
                      int32_t dim)
    {
        int32_t m = v % dim;
        return (m < 0) ? (m + dim) : m;
    }

    int32_t Flatten(int32_t x,
                    int32_t y,
                    int32_t z,
                    int32_t ny,
                    int32_t nz)
    {
        return (x * ny + y) * nz + z;
    }
}

using namespace SlimeMold3DKernelTestsInternal;

// With a flat (zero) field and no deposit and no wander, an agent has nothing
// to sense or be pushed by, so every step is a deterministic straight move
// along its (unchanged) heading. The GPU result must match that closed-form
// advance — a direct check of the move + wrap math and the agent readback,
// isolated from the chaotic sense/steer coupling the showcase relies on.
TEST_CASE("Hal::Gpu::SlimeMold3DKernel advances agents straight under a flat field, or reports why it can't")
{
    try
    {
        Hal::Gpu::ComputeContext context;

        const int32_t n = 24;
        const int32_t agentCount = 64;
        const int32_t steps = 5;

        Hal::Gpu::SlimeMold3DKernel kernel(context, n, n, n, agentCount);

        // Deterministic scattered agents near the center with normalized
        // headings, kept away from the wrap seams so the closed-form advance
        // stays inside the domain.
        std::vector<float> agents(static_cast<size_t>(agentCount) * 8);
        std::vector<double> hx(static_cast<size_t>(agentCount));
        std::vector<double> hy(static_cast<size_t>(agentCount));
        std::vector<double> hz(static_cast<size_t>(agentCount));
        std::vector<double> px(static_cast<size_t>(agentCount));
        std::vector<double> py(static_cast<size_t>(agentCount));
        std::vector<double> pz(static_cast<size_t>(agentCount));
        for (int32_t i = 0; i < agentCount; i++)
        {
            double dx = std::sin(0.7 * i + 0.1);
            double dy = std::cos(0.4 * i + 1.3);
            double dz = std::sin(0.9 * i + 2.0) * 0.5;
            double len = std::sqrt(dx * dx + dy * dy + dz * dz);
            dx /= len;
            dy /= len;
            dz /= len;
            double x = 12.0 + std::fmod(0.11 * i, 1.0);
            double y = 12.0 + std::fmod(0.23 * i, 1.0);
            double z = 12.0 + std::fmod(0.31 * i, 1.0);

            hx[static_cast<size_t>(i)] = dx;
            hy[static_cast<size_t>(i)] = dy;
            hz[static_cast<size_t>(i)] = dz;
            px[static_cast<size_t>(i)] = x;
            py[static_cast<size_t>(i)] = y;
            pz[static_cast<size_t>(i)] = z;

            agents[static_cast<size_t>(8 * i)] = static_cast<float>(x);
            agents[static_cast<size_t>(8 * i + 1)] = static_cast<float>(y);
            agents[static_cast<size_t>(8 * i + 2)] = static_cast<float>(z);
            agents[static_cast<size_t>(8 * i + 3)] = 0.0f;
            agents[static_cast<size_t>(8 * i + 4)] = static_cast<float>(dx);
            agents[static_cast<size_t>(8 * i + 5)] = static_cast<float>(dy);
            agents[static_cast<size_t>(8 * i + 6)] = static_cast<float>(dz);
            agents[static_cast<size_t>(8 * i + 7)] = 0.0f;
        }

        Hal::Gpu::SlimeMold3DParams params = {};
        params.stepSize = 0.7f;
        params.sensorAngle = 0.5f;
        params.sensorDist = 3.0f;
        params.turnRate = 0.3f;
        params.depositAmount = 0.0f; // no deposit -> field stays flat/zero
        params.jitter = 0.0f;        // no wander -> heading is preserved exactly
        params.diffuse = 0.1f;
        params.decay = 0.9f;
        params.seed = 12345u;

        kernel.RunSteps(agents, agentCount, steps, params);

        // Closed-form straight advance (per-step wrap, matching the shader).
        double maxErr = 0.0;
        for (int32_t i = 0; i < agentCount; i++)
        {
            double x = px[static_cast<size_t>(i)];
            double y = py[static_cast<size_t>(i)];
            double z = pz[static_cast<size_t>(i)];
            for (int32_t s = 0; s < steps; s++)
            {
                x = std::fmod(x + hx[static_cast<size_t>(i)] * params.stepSize + n, static_cast<double>(n));
                y = std::fmod(y + hy[static_cast<size_t>(i)] * params.stepSize + n, static_cast<double>(n));
                z = std::fmod(z + hz[static_cast<size_t>(i)] * params.stepSize + n, static_cast<double>(n));
            }
            maxErr = std::max(maxErr, std::abs(static_cast<double>(agents[static_cast<size_t>(8 * i)]) - x));
            maxErr = std::max(maxErr, std::abs(static_cast<double>(agents[static_cast<size_t>(8 * i + 1)]) - y));
            maxErr = std::max(maxErr, std::abs(static_cast<double>(agents[static_cast<size_t>(8 * i + 2)]) - z));
        }
        CHECK(maxErr < 1e-3);
    }
    catch (const std::runtime_error &e)
    {
        MESSAGE("No usable Vulkan compute device in this environment: ", std::string(e.what()));
    }
}

// The field pass is a deterministic 7-point diffuse + decay. Run it alone
// (zero agents, so no deposits perturb it) on a seeded field and check it
// against a bit-identical CPU mirror of the same stencil — the plumbing check
// (ping-pong, barriers, seed/read round-trip) an analytic mode can't give a
// coupled system.
TEST_CASE("Hal::Gpu::SlimeMold3DKernel field pass matches a CPU 7-point diffuse+decay mirror, or reports why it can't")
{
    try
    {
        Hal::Gpu::ComputeContext context;

        const int32_t n = 16;
        const int32_t length = n * n * n;
        const int32_t steps = 4;
        const float diffuse = 0.12f;
        const float decay = 0.9f;

        Hal::Gpu::SlimeMold3DKernel kernel(context, n, n, n, 1);

        // Deterministic seed field in [0, 1].
        std::vector<float> field(static_cast<size_t>(length));
        for (int32_t i = 0; i < length; i++)
        {
            field[static_cast<size_t>(i)] = 0.5f + 0.5f * std::sin(0.37f * static_cast<float>(i));
        }

        std::vector<float> cpu = field;
        std::vector<float> next(static_cast<size_t>(length));
        for (int32_t s = 0; s < steps; s++)
        {
            for (int32_t x = 0; x < n; x++)
            {
                for (int32_t y = 0; y < n; y++)
                {
                    for (int32_t z = 0; z < n; z++)
                    {
                        int32_t xm = WrapIndex(x - 1, n);
                        int32_t xp = WrapIndex(x + 1, n);
                        int32_t ym = WrapIndex(y - 1, n);
                        int32_t yp = WrapIndex(y + 1, n);
                        int32_t zm = WrapIndex(z - 1, n);
                        int32_t zp = WrapIndex(z + 1, n);
                        float c = cpu[static_cast<size_t>(Flatten(x, y, z, n, n))];
                        float neighbors =
                            cpu[static_cast<size_t>(Flatten(xm, y, z, n, n))] + cpu[static_cast<size_t>(Flatten(xp, y, z, n, n))] +
                            cpu[static_cast<size_t>(Flatten(x, ym, z, n, n))] + cpu[static_cast<size_t>(Flatten(x, yp, z, n, n))] +
                            cpu[static_cast<size_t>(Flatten(x, y, zm, n, n))] + cpu[static_cast<size_t>(Flatten(x, y, zp, n, n))];
                        float diffused = c + diffuse * (neighbors - 6.0f * c);
                        float value = diffused * decay;
                        next[static_cast<size_t>(Flatten(x, y, z, n, n))] = (value > 0.0f) ? value : 0.0f;
                    }
                }
            }
            cpu.swap(next);
        }

        kernel.SeedField(field);
        std::vector<float> empty;
        Hal::Gpu::SlimeMold3DParams params = {};
        params.stepSize = 1.0f;
        params.sensorAngle = 0.5f;
        params.sensorDist = 3.0f;
        params.turnRate = 0.3f;
        params.depositAmount = 0.0f;
        params.jitter = 0.0f;
        params.diffuse = diffuse;
        params.decay = decay;
        params.seed = 7u;
        kernel.RunSteps(empty, 0, steps, params);

        std::vector<float> gpu;
        kernel.ReadField(gpu);

        REQUIRE(gpu.size() == static_cast<size_t>(length));
        float maxErr = 0.0f;
        for (int32_t i = 0; i < length; i++)
        {
            maxErr = std::max(maxErr, std::abs(gpu[static_cast<size_t>(i)] - cpu[static_cast<size_t>(i)]));
        }
        CHECK(maxErr < 1e-4f);
    }
    catch (const std::runtime_error &e)
    {
        MESSAGE("No usable Vulkan compute device in this environment: ", std::string(e.what()));
    }
}
