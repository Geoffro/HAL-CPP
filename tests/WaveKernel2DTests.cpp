#include <doctest/doctest.h>

#include "GpuTestSupport.h"

#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/WaveKernel2D.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

// A single periodic spatial eigenmode makes the leapfrog wave update an
// EXACT two-term recurrence, giving a closed-form ground truth (no
// CPU-vs-GPU comparison, so a bug shared by both couldn't hide).
//
// For phi(x,y) = sin(2*pi*p*x/nx) * sin(2*pi*q*y/ny), the periodic 5-point
// Laplacian has eigenvalue lambda = -4*(sin^2(pi*p/nx) + sin^2(pi*q/ny)).
// The update u_next = 2*u - u_prev + speed2*lambda*u is then
// u_{n+1} = 2*cos(theta)*u_n - u_{n-1} with cos(theta) = 1 + speed2*lambda/2,
// whose exact solution with u_0 = phi, u_{-1} = cos(theta)*phi is
// u_n = cos(n*theta)*phi. With damping = 1 (no energy loss) the GPU field
// after N steps must equal cos(N*theta)*phi to float precision.
TEST_CASE("Hal::Gpu::WaveKernel2D reproduces the exact standing-wave eigenmode recurrence")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::ComputeContext &context = *pContext;

    const int32_t nx = 32;
    const int32_t ny = 48;
    const int32_t p = 8;
    const int32_t q = 12;
    const int32_t steps = 40;
    const float speed2 = 0.3f;

    Hal::Gpu::WaveKernel2D kernel(context, nx, ny);

    std::vector<float> phi(static_cast<size_t>(nx * ny));
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            phi[static_cast<size_t>(x * ny + y)] = static_cast<float>(std::sin(2.0 * M_PI * p * x / nx) * std::sin(2.0 * M_PI * q * y / ny));
        }
    }

    double lambda = -4.0 * (std::pow(std::sin(M_PI * p / nx), 2) + std::pow(std::sin(M_PI * q / ny), 2));
    double cosTheta = 1.0 + speed2 * lambda / 2.0;
    double theta = std::acos(cosTheta);

    std::vector<float> cur = phi;
    std::vector<float> prev(static_cast<size_t>(nx * ny));
    for (size_t i = 0; i < prev.size(); i++)
    {
        prev[i] = static_cast<float>(cosTheta * phi[i]);
    }

    kernel.RunSteps(cur, prev, steps, speed2, 1.0f, true);

    double amp = std::cos(steps * theta);
    double maxErr = 0.0;
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            size_t i = static_cast<size_t>(x * ny + y);
            double expected = amp * std::sin(2.0 * M_PI * p * x / nx) * std::sin(2.0 * M_PI * q * y / ny);
            maxErr = std::max(maxErr, std::abs(static_cast<double>(cur[i]) - expected));
        }
    }
    CHECK(maxErr < 2e-3);

    // One more batch continues the same recurrence (2N total),
    // exercising staging/command-buffer reuse and the prev/cur handoff.
    kernel.RunSteps(cur, prev, steps, speed2, 1.0f, true);
    double amp2 = std::cos(2 * steps * theta);
    double probe = static_cast<double>(cur[static_cast<size_t>(3 * ny + 5)]);
    double expected2 = amp2 * std::sin(2.0 * M_PI * p * 3 / nx) * std::sin(2.0 * M_PI * q * 5 / ny);
    CHECK(probe == doctest::Approx(expected2).epsilon(5e-3).scale(1.0));
}

TEST_CASE("Hal::Gpu::WaveKernel2D rejects current or previous fields of the wrong length")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::WaveKernel2D kernel(*pContext, 4, 4);
    std::vector<float> rightSize(16, 0.0f);
    std::vector<float> wrongSize(15, 0.0f);
    CHECK_THROWS_AS(kernel.RunSteps(wrongSize, rightSize, 1, 0.25f, 1.0f, true), std::invalid_argument);
    CHECK_THROWS_AS(kernel.RunSteps(rightSize, wrongSize, 1, 0.25f, 1.0f, true), std::invalid_argument);
}
