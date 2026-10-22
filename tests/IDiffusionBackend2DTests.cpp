#include <doctest/doctest.h>

#include "GpuTestSupport.h"

#include "Hal/Pde/CpuAdiDiffusionBackend2D.h"
#include "Hal/Pde/CpuDiffusionBackend2D.h"
#include "Hal/Pde/IAdiDiffusionBackend2D.h"
#include "Hal/Pde/IDiffusionBackend2D.h"
#include "Hal/Gpu/AdiKernel2D.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/GpuAdiDiffusionBackend2D.h"
#include "Hal/Gpu/GpuDiffusionBackend2D.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

// One call site run against each backend through the interface, checked
// against the analytic eigenmode decay rather than backend against
// backend.
static void RunEigenmodeCheck(Hal::IDiffusionBackend2D &backend)
{
    const int32_t nx = 16;
    const int32_t ny = 20;
    const double diffRate = 0.05;

    std::vector<double> field(static_cast<size_t>(nx * ny));
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            field[static_cast<size_t>(x * ny + y)] = std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny);
        }
    }
    std::vector<double> deltas(field.size(), 0.0);

    backend.Diffuse(field, deltas, diffRate, nx, ny, true, true);

    double lambda = 1.0 - 4.0 * diffRate * (std::pow(std::sin(M_PI / nx), 2) + std::pow(std::sin(M_PI / ny), 2));
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            size_t i = static_cast<size_t>(x * ny + y);
            double expected = lambda * std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny);
            double actual = field[i] + deltas[i];
            CAPTURE(x);
            CAPTURE(y);
            CHECK(actual == doctest::Approx(expected).epsilon(1e-4));
        }
    }
}

TEST_CASE("Hal::CpuDiffusionBackend2D matches the analytic eigenmode decay factor through the IDiffusionBackend2D interface")
{
    Hal::CpuDiffusionBackend2D backend;
    RunEigenmodeCheck(backend);
}

TEST_CASE("Hal::Gpu::GpuDiffusionBackend2D matches the analytic eigenmode decay factor through the same IDiffusionBackend2D interface")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::ComputeContext &context = *pContext;
    Hal::Gpu::GpuDiffusionBackend2D backend(context, 16, 20);
    RunEigenmodeCheck(backend);
}

// The ADI interface, checked against the analytic per-axis Crank-Nicolson
// factor product (as in PDEADITests.cpp).
static void RunAdiEigenmodeCheck(Hal::IAdiDiffusionBackend2D &backend,
                                 double                       epsilon)
{
    const int32_t nx = 12;
    const int32_t ny = 18;
    const double diffRate = 0.3;

    std::vector<double> field(static_cast<size_t>(nx * ny));
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            field[static_cast<size_t>(x * ny + y)] = std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny);
        }
    }
    std::vector<double> deltas(field.size(), 0.0);

    backend.DiffuseADI(field, deltas, diffRate, nx, ny, true, true);

    double thetaX = 2.0 * M_PI / nx;
    double thetaY = 2.0 * M_PI / ny;
    double sX = std::pow(std::sin(thetaX / 2.0), 2);
    double sY = std::pow(std::sin(thetaY / 2.0), 2);
    double lambda = ((1.0 - 2.0 * diffRate * sX) / (1.0 + 2.0 * diffRate * sX)) *
                    ((1.0 - 2.0 * diffRate * sY) / (1.0 + 2.0 * diffRate * sY));
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            size_t i = static_cast<size_t>(x * ny + y);
            double expected = lambda * std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny);
            double actual = field[i] + deltas[i];
            CAPTURE(x);
            CAPTURE(y);
            CHECK(actual == doctest::Approx(expected).epsilon(epsilon).scale(1.0));
        }
    }
}

TEST_CASE("Hal::CpuAdiDiffusionBackend2D matches the analytic ADI eigenmode decay through the strategy interface")
{
    Hal::CpuAdiDiffusionBackend2D backend;
    RunAdiEigenmodeCheck(backend, 1e-7);

    // A second call through the same backend instance (scratch reuse
    // path) gives the same answer.
    RunAdiEigenmodeCheck(backend, 1e-7);
}

// The GPU ADI backend (PCR tridiagonal solve per line — see
// shaders/AdiSweep2D.comp) through the exact same call site, against the
// same analytic ground truth, at float precision.
TEST_CASE("Hal::Gpu::GpuAdiDiffusionBackend2D matches the analytic ADI eigenmode decay through the same call site")
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
    Hal::Gpu::GpuAdiDiffusionBackend2D backend(context, 12, 18);
    RunAdiEigenmodeCheck(backend, 1e-4);
    RunAdiEigenmodeCheck(backend, 1e-4);
}

TEST_CASE("CPU diffusion backends reject field and delta arrays that don't match the dimensions")
{
    Hal::CpuDiffusionBackend2D explicitBackend;
    Hal::CpuAdiDiffusionBackend2D adiBackend;
    std::vector<double> field(12, 0.0);
    std::vector<double> deltas(11, 0.0);
    CHECK_THROWS_AS(explicitBackend.Diffuse(field, deltas, 0.1, 3, 4, false, false), std::invalid_argument);
    CHECK_THROWS_AS(adiBackend.DiffuseADI(field, deltas, 0.1, 3, 4, false, false), std::invalid_argument);
    deltas.resize(12);
    CHECK_THROWS_AS(explicitBackend.Diffuse(field, deltas, 0.1, 4, 4, false, false), std::invalid_argument);
    CHECK_NOTHROW(adiBackend.DiffuseADI(field, deltas, 0.1, 3, 4, false, false));
}

TEST_CASE("GPU diffusion backends reject field and delta arrays that don't match the dimensions")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::GpuDiffusionBackend2D explicitBackend(*pContext, 3, 4);
    std::vector<double> field(12, 0.0);
    std::vector<double> deltas(11, 0.0);
    CHECK_THROWS_AS(explicitBackend.Diffuse(field, deltas, 0.1, 3, 4, false, false), std::invalid_argument);
    if (!Hal::Gpu::AdiKernel2D::IsSupported(*pContext))
    {
        MESSAGE("Skipped the ADI backend: device workgroup limits are too small for AdiKernel2D");
        return;
    }
    Hal::Gpu::GpuAdiDiffusionBackend2D adiBackend(*pContext, 3, 4);
    CHECK_THROWS_AS(adiBackend.DiffuseADI(field, deltas, 0.1, 3, 4, false, false), std::invalid_argument);
}
