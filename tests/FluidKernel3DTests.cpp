#include <doctest/doctest.h>

#include "GpuTestSupport.h"

#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/FluidKernel3D.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace FluidTestHelpers
{
    const int32_t g_xDim = 8;
    const int32_t g_yDim = 12;
    const int32_t g_zDim = 6;

    // No dissipation, no source, no buoyancy.
    Hal::Gpu::FluidParams StillParams()
    {
        Hal::Gpu::FluidParams params  = {};
        params.dt                     = 1.0f;
        params.velocityDissipation    = 1.0f;
        params.densityDissipation     = 1.0f;
        params.temperatureDissipation = 1.0f;
        params.buoyancy               = 0.0f;
        params.ambientTemperature     = 0.0f;
        params.sourceStrength         = 0.0f;
        params.pressureIterations     = 20;
        return params;
    }

    // Interleaved (density, temperature) with a hot, dense ball near the
    // bottom of the box.
    std::vector<float> HotBlob()
    {
        std::vector<float> scalars(static_cast<size_t>(g_xDim * g_yDim * g_zDim) * 2, 0.0f);
        for (int32_t x = 0; x < g_xDim; x++)
        {
            for (int32_t y = 0; y < g_yDim; y++)
            {
                for (int32_t z = 0; z < g_zDim; z++)
                {
                    double dx = x - 3.5;
                    double dy = y - 3.0;
                    double dz = z - 2.5;
                    if (((dx * dx) + (dy * dy) + (dz * dz)) < 4.0)
                    {
                        size_t i = static_cast<size_t>((x * g_yDim + y) * g_zDim + z);
                        scalars[i * 2] = 1.0f;
                        scalars[i * 2 + 1] = 1.0f;
                    }
                }
            }
        }
        return scalars;
    }

    // Temperature-weighted mean y.
    double MeanHotY(const std::vector<float> &scalars)
    {
        double weighted = 0.0;
        double total = 0.0;
        for (int32_t x = 0; x < g_xDim; x++)
        {
            for (int32_t y = 0; y < g_yDim; y++)
            {
                for (int32_t z = 0; z < g_zDim; z++)
                {
                    size_t i = static_cast<size_t>((x * g_yDim + y) * g_zDim + z);
                    weighted += scalars[i * 2 + 1] * y;
                    total += scalars[i * 2 + 1];
                }
            }
        }
        return weighted / total;
    }

    void CheckSame(const std::vector<float> &a,
                   const std::vector<float> &b)
    {
        REQUIRE(a.size() == b.size());
        for (size_t i = 0; i < a.size(); i++)
        {
            CAPTURE(i);
            CHECK(a[i] == doctest::Approx(b[i]).epsilon(1e-5).scale(1.0));
        }
    }
}

using FluidTestHelpers::g_xDim;
using FluidTestHelpers::g_yDim;
using FluidTestHelpers::g_zDim;

TEST_CASE("Hal::Gpu::FluidKernel3D leaves a uniform, still field unchanged")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::FluidKernel3D kernel(*pContext, g_xDim, g_yDim, g_zDim);
    std::vector<float> scalars(static_cast<size_t>(g_xDim * g_yDim * g_zDim) * 2);
    for (size_t i = 0; i < scalars.size(); i += 2)
    {
        scalars[i] = 0.7f;
        scalars[i + 1] = 0.0f;
    }
    std::vector<float> original = scalars;

    Hal::Gpu::FluidParams params = FluidTestHelpers::StillParams();
    params.buoyancy = 0.5f;
    kernel.RunSteps(scalars, 5, params);
    FluidTestHelpers::CheckSame(scalars, original);
}

TEST_CASE("Hal::Gpu::FluidKernel3D buoyancy lifts hot fluid")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::FluidKernel3D kernel(*pContext, g_xDim, g_yDim, g_zDim);
    std::vector<float> scalars = FluidTestHelpers::HotBlob();
    double before = FluidTestHelpers::MeanHotY(scalars);

    Hal::Gpu::FluidParams params = FluidTestHelpers::StillParams();
    params.buoyancy = 0.3f;
    kernel.RunSteps(scalars, 10, params);
    CHECK(FluidTestHelpers::MeanHotY(scalars) > (before + 0.5));
}

TEST_CASE("Hal::Gpu::FluidKernel3D carries its velocity field across RunSteps calls")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::FluidParams params = FluidTestHelpers::StillParams();
    params.buoyancy = 0.3f;
    params.pressureIterations = 6;

    Hal::Gpu::FluidKernel3D batched(*pContext, g_xDim, g_yDim, g_zDim);
    std::vector<float> batchedScalars = FluidTestHelpers::HotBlob();
    batched.RunSteps(batchedScalars, 3, params);

    Hal::Gpu::FluidKernel3D stepped(*pContext, g_xDim, g_yDim, g_zDim);
    std::vector<float> steppedScalars = FluidTestHelpers::HotBlob();
    for (int32_t step = 0; step < 3; step++)
    {
        stepped.RunSteps(steppedScalars, 1, params);
    }
    FluidTestHelpers::CheckSame(steppedScalars, batchedScalars);

    // An odd pressure iteration count is rounded up to the next even one.
    params.pressureIterations = 5;
    Hal::Gpu::FluidKernel3D rounded(*pContext, g_xDim, g_yDim, g_zDim);
    std::vector<float> roundedScalars = FluidTestHelpers::HotBlob();
    rounded.RunSteps(roundedScalars, 3, params);
    FluidTestHelpers::CheckSame(roundedScalars, batchedScalars);
}

TEST_CASE("Hal::Gpu::FluidKernel3D rejects a scalar array of the wrong length")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::FluidKernel3D kernel(*pContext, g_xDim, g_yDim, g_zDim);
    std::vector<float> scalars(static_cast<size_t>(g_xDim * g_yDim * g_zDim), 0.0f);
    CHECK_THROWS_AS(kernel.RunSteps(scalars, 1, FluidTestHelpers::StillParams()), std::invalid_argument);
}
