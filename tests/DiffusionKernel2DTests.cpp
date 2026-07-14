#include <doctest/doctest.h>

#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/DiffusionKernel2D.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// Reuses the exact same closed-form eigenmode-decay check as
// PDEGridTests.cpp's "PDEGrid2D::Diffusion exactly matches the analytic
// eigenmode decay factor" case, verifying the GPU stencil (Diffusion2D.comp)
// against the same ground truth rather than against the CPU path (so a
// shared bug in both wouldn't hide behind an agreement of the two).
//
// Skipped (via MESSAGE, not a failing CHECK) if this environment has no
// usable Vulkan compute device, matching ComputeContextTests.cpp's approach.
TEST_CASE("Hal::Gpu::DiffusionKernel2D exactly matches the analytic eigenmode decay factor (periodic, product mode)")
{
    try
    {
        Hal::Gpu::ComputeContext context;

        const int32_t nx = 16;
        const int32_t ny = 20;
        const double diffRate = 0.05;

        Hal::Gpu::DiffusionKernel2D kernel(context, nx, ny);

        std::vector<float> field(static_cast<size_t>(nx * ny));
        for (int32_t x = 0; x < nx; x++)
        {
            for (int32_t y = 0; y < ny; y++)
            {
                field[static_cast<size_t>(x * ny + y)] = static_cast<float>(std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny));
            }
        }

        kernel.UploadField(field);
        kernel.ClearDeltas();
        kernel.Dispatch(static_cast<float>(diffRate), true, true);
        std::vector<float> deltas = kernel.DownloadDeltas();

        double lambda = 1.0 - 4.0 * diffRate * (std::pow(std::sin(M_PI / nx), 2) + std::pow(std::sin(M_PI / ny), 2));
        for (int32_t x = 0; x < nx; x++)
        {
            for (int32_t y = 0; y < ny; y++)
            {
                size_t i = static_cast<size_t>(x * ny + y);
                double expected = lambda * std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny);
                double actual = static_cast<double>(field[i]) + static_cast<double>(deltas[i]);
                CAPTURE(x);
                CAPTURE(y);
                CHECK(actual == doctest::Approx(expected).epsilon(1e-5));
            }
        }
    }
    catch (const std::runtime_error &e)
    {
        MESSAGE("No usable Vulkan compute device in this environment: ", std::string(e.what()));
    }
}

// The GPU-resident multi-step path: RunSteps chains N (stencil + commit)
// steps inside one submission, so after N steps a periodic eigenmode must
// be scaled by exactly lambda^N — a ground truth no pairwise CPU/GPU
// comparison could provide, and one that fails if the DiffusionUpdate2D
// commit pass or any inter-pass barrier is wrong.
TEST_CASE("Hal::Gpu::DiffusionKernel2D::RunSteps matches the analytic eigenmode decay to the Nth power, or reports why it can't")
{
    try
    {
        Hal::Gpu::ComputeContext context;

        const int32_t nx = 16;
        const int32_t ny = 20;
        const int32_t steps = 50;
        const double diffRate = 0.05;

        Hal::Gpu::DiffusionKernel2D kernel(context, nx, ny);

        std::vector<float> field(static_cast<size_t>(nx * ny));
        for (int32_t x = 0; x < nx; x++)
        {
            for (int32_t y = 0; y < ny; y++)
            {
                field[static_cast<size_t>(x * ny + y)] = static_cast<float>(std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny));
            }
        }

        kernel.RunSteps(field, steps, static_cast<float>(diffRate), true, true);

        double lambda = 1.0 - 4.0 * diffRate * (std::pow(std::sin(M_PI / nx), 2) + std::pow(std::sin(M_PI / ny), 2));
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

        // A second RunSteps on the evolved field continues the decay
        // (2N total), covering staging/command-buffer reuse.
        kernel.RunSteps(field, steps, static_cast<float>(diffRate), true, true);
        double lambda2N = std::pow(lambda, 2 * steps);
        CHECK(field[static_cast<size_t>(1 * ny + 1)] == doctest::Approx(lambda2N * std::sin(2.0 * M_PI / nx) * std::sin(2.0 * M_PI / ny)).epsilon(2e-3).scale(1.0));
    }
    catch (const std::runtime_error &e)
    {
        MESSAGE("No usable Vulkan compute device in this environment: ", std::string(e.what()));
    }
}
