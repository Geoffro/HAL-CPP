#include <doctest/doctest.h>

#include "GpuTestSupport.h"

#include "Hal/Gpu/AdiKernel2D.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Pde/CpuAdiDiffusionBackend2D.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

// After N steps a periodic eigenmode must be scaled by exactly
// (lambdaX * lambdaY)^N, each lambda the per-axis Crank-Nicolson factor
// (the ground truth PDEADITests.cpp uses for the CPU solver).
TEST_CASE("Hal::Gpu::AdiKernel2D::RunSteps matches the analytic Crank-Nicolson eigenmode decay to the Nth power")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::ComputeContext &context = *pContext;
    if (!Hal::Gpu::AdiKernel2D::IsSupported(context))
    {
        MESSAGE("Skipped: device workgroup limits are too small for AdiKernel2D");
        return;
    }

    const int32_t nx = 24;
    const int32_t ny = 16;
    const int32_t steps = 50;
    // Deliberately above the explicit scheme's 0.25 CFL cap: ADI has
    // no such limit, and this confirms the GPU solve is genuinely
    // implicit.
    const double diffRate = 0.8;

    Hal::Gpu::AdiKernel2D kernel(context, nx, ny);

    std::vector<float> field(static_cast<size_t>(nx * ny));
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            field[static_cast<size_t>(x * ny + y)] = static_cast<float>(std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny));
        }
    }

    kernel.RunSteps(field, steps, static_cast<float>(diffRate), true, true);

    double sX = std::pow(std::sin(M_PI / nx), 2);
    double sY = std::pow(std::sin(M_PI / ny), 2);
    double lambda = ((1.0 - 2.0 * diffRate * sX) / (1.0 + 2.0 * diffRate * sX)) *
                    ((1.0 - 2.0 * diffRate * sY) / (1.0 + 2.0 * diffRate * sY));
    double lambdaN = std::pow(lambda, steps);
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            double expected = lambdaN * std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny);
            CAPTURE(x);
            CAPTURE(y);
            CHECK(field[static_cast<size_t>(x * ny + y)] == doctest::Approx(expected).epsilon(1e-3).scale(1.0));
        }
    }
}

TEST_CASE("Hal::Gpu::AdiKernel2D zero-flux run conserves mass and matches the CPU ADI solver cell-for-cell")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::ComputeContext &context = *pContext;
    if (!Hal::Gpu::AdiKernel2D::IsSupported(context))
    {
        MESSAGE("Skipped: device workgroup limits are too small for AdiKernel2D");
        return;
    }

    const int32_t nx = 20;
    const int32_t ny = 14;
    const int32_t steps = 5;
    const double diffRate = 0.6;

    // A non-eigenmode field so agreement isn't an artifact of
    // symmetry.
    std::vector<double> cpuField(static_cast<size_t>(nx * ny));
    std::vector<float> gpuField(static_cast<size_t>(nx * ny));
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            double value = 1.0 + 0.3 * x - 0.2 * y + 0.5 * std::sin(0.7 * x * y);
            cpuField[static_cast<size_t>(x * ny + y)] = value;
            gpuField[static_cast<size_t>(x * ny + y)] = static_cast<float>(value);
        }
    }

    double massBefore = 0.0;
    for (double v : cpuField)
    {
        massBefore += v;
    }

    // CPU reference: N steps of DiffuseADI + commit.
    Hal::CpuAdiDiffusionBackend2D cpuBackend;
    std::vector<double> deltas(cpuField.size(), 0.0);
    for (int32_t step = 0; step < steps; step++)
    {
        cpuBackend.DiffuseADI(cpuField, deltas, diffRate, nx, ny, false, false);
        for (size_t k = 0; k < cpuField.size(); k++)
        {
            cpuField[k] += deltas[k];
            deltas[k] = 0.0;
        }
    }

    Hal::Gpu::AdiKernel2D kernel(context, nx, ny);
    kernel.RunSteps(gpuField, steps, static_cast<float>(diffRate), false, false);

    double massAfter = 0.0;
    for (float v : gpuField)
    {
        massAfter += static_cast<double>(v);
    }
    CHECK(massAfter == doctest::Approx(massBefore).epsilon(1e-4));

    for (size_t k = 0; k < cpuField.size(); k++)
    {
        CAPTURE(k);
        CHECK(gpuField[k] == doctest::Approx(cpuField[k]).epsilon(1e-4).scale(1.0));
    }

    // Oversized grids are rejected up front (the PCR line must fit
    // one workgroup's shared memory).
    CHECK_THROWS_AS(Hal::Gpu::AdiKernel2D(context, Hal::Gpu::AdiKernel2D::kMaxLine + 1, 8), std::invalid_argument);
}

TEST_CASE("Hal::Gpu::AdiKernel2D rejects a field of the wrong length")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    if (!Hal::Gpu::AdiKernel2D::IsSupported(*pContext))
    {
        MESSAGE("Skipped: device workgroup limits are too small for AdiKernel2D");
        return;
    }
    Hal::Gpu::AdiKernel2D kernel(*pContext, 4, 4);
    std::vector<float> field(15, 0.0f);
    CHECK_THROWS_AS(kernel.RunSteps(field, 1, 0.1f, false, false), std::invalid_argument);
}
