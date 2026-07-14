#include <doctest/doctest.h>

#include "Hal/Pde/CpuAdiDiffusionBackend2D.h"
#include "Hal/Pde/CpuDiffusionBackend2D.h"
#include "Hal/Pde/IAdiDiffusionBackend2D.h"
#include "Hal/Pde/IDiffusionBackend2D.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/GpuAdiDiffusionBackend2D.h"
#include "Hal/Gpu/GpuDiffusionBackend2D.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// Exercises Hal::IDiffusionBackend2D as the strategy interface it's meant
// to be: the same call site (RunEigenmodeCheck below) runs unmodified
// against whichever concrete backend it's handed, confirming the
// docs/PHASED_PORT.md "IComputeBackend" milestone's actual point — model
// code doesn't need to know which backend it's using. Reuses the same
// analytic eigenmode-decay ground truth as PDEGridTests.cpp/
// DiffusionKernel2DTests.cpp rather than comparing backends to each
// other, so a bug shared by both implementations wouldn't hide behind
// their agreement.
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
    try
    {
        Hal::Gpu::ComputeContext context;
        Hal::Gpu::GpuDiffusionBackend2D backend(context, 16, 20);
        RunEigenmodeCheck(backend);
    }
    catch (const std::runtime_error &e)
    {
        MESSAGE("No usable Vulkan compute device in this environment: ", std::string(e.what()));
    }
}

// The ADI strategy interface, checked the same way against the analytic
// per-axis Crank-Nicolson amplification-factor product (the ground truth
// PDEADITests.cpp uses) rather than against PDEGrid2D::DiffusionADI —
// they share internals, so agreement with the grid method alone wouldn't
// prove correctness.
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
TEST_CASE("Hal::Gpu::GpuAdiDiffusionBackend2D matches the analytic ADI eigenmode decay through the same call site, or reports why it can't")
{
    try
    {
        Hal::Gpu::ComputeContext context;
        Hal::Gpu::GpuAdiDiffusionBackend2D backend(context, 12, 18);
        RunAdiEigenmodeCheck(backend, 1e-4);
        RunAdiEigenmodeCheck(backend, 1e-4);
    }
    catch (const std::runtime_error &e)
    {
        MESSAGE("No usable Vulkan compute device in this environment: ", std::string(e.what()));
    }
}
