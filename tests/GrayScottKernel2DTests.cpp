#include <doctest/doctest.h>

#include "GpuTestSupport.h"

#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/GrayScottKernel2D.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace GrayScottKernel2DTestsInternal
{
    int32_t WrapIndex(int32_t v,
                      int32_t dim)
    {
        int32_t m = v % dim;
        return (m < 0) ? (m + dim) : m;
    }

    // CPU mirror of shaders/GrayScott2D.comp (same Laplacian, reaction,
    // clamp, wrap, and ping-pong order) for checking the full nonlinear
    // dynamics.
    void CpuGrayScottStep(const std::vector<float> &src,
                          std::vector<float>       &dst,
                          int32_t                   xDim,
                          int32_t                   yDim,
                          float                     feed,
                          float                     kill,
                          float                     diffU,
                          float                     diffV)
    {
        for (int32_t x = 0; x < xDim; x++)
        {
            for (int32_t y = 0; y < yDim; y++)
            {
                int32_t xm = WrapIndex(x - 1, xDim);
                int32_t xp = WrapIndex(x + 1, xDim);
                int32_t ym = WrapIndex(y - 1, yDim);
                int32_t yp = WrapIndex(y + 1, yDim);

                size_t c = static_cast<size_t>(x * yDim + y);
                float u = src[2 * c];
                float v = src[2 * c + 1];

                for (int32_t comp = 0; comp < 2; comp++)
                {
                    float orthogonal = src[2 * static_cast<size_t>(xm * yDim + y) + comp] + src[2 * static_cast<size_t>(xp * yDim + y) + comp] + src[2 * static_cast<size_t>(x * yDim + ym) + comp] + src[2 * static_cast<size_t>(x * yDim + yp) + comp];
                    float diagonal = src[2 * static_cast<size_t>(xm * yDim + ym) + comp] + src[2 * static_cast<size_t>(xm * yDim + yp) + comp] + src[2 * static_cast<size_t>(xp * yDim + ym) + comp] + src[2 * static_cast<size_t>(xp * yDim + yp) + comp];
                    float center = src[2 * c + comp];
                    float lap = 0.2f * orthogonal + 0.05f * diagonal - center;

                    float reaction = u * v * v;
                    if (comp == 0)
                    {
                        float uNext = u + (diffU * lap - reaction + feed * (1.0f - u));
                        dst[2 * c] = std::min(std::max(uNext, 0.0f), 1.0f);
                    }
                    else
                    {
                        float vNext = v + (diffV * lap + reaction - (feed + kill) * v);
                        dst[2 * c + 1] = std::min(std::max(vNext, 0.0f), 1.0f);
                    }
                }
            }
        }
    }
}

using namespace GrayScottKernel2DTestsInternal;

// (U=1, V=0) everywhere is an exact fixed point of Gray-Scott: the
// reaction term u*v*v is zero, the Laplacian of a constant is zero, and
// feed*(1-U) is zero at U=1. RunSteps must leave it untouched — a cheap
// analytic invariant that would break if the reaction sign, the feed term,
// or the buffer plumbing were wrong.
TEST_CASE("Hal::Gpu::GrayScottKernel2D leaves the (U=1, V=0) fixed point unchanged")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::ComputeContext &context = *pContext;

    const int32_t nx = 32;
    const int32_t ny = 48;

    Hal::Gpu::GrayScottKernel2D kernel(context, nx, ny);

    std::vector<float> u(static_cast<size_t>(nx * ny), 1.0f);
    std::vector<float> v(static_cast<size_t>(nx * ny), 0.0f);

    kernel.RunSteps(u, v, 25, 0.055f, 0.062f, 0.16f, 0.08f);

    for (size_t i = 0; i < u.size(); i++)
    {
        CHECK(u[i] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(v[i] == doctest::Approx(0.0).epsilon(1e-6));
    }
}

// The full nonlinear dynamics, checked against the bit-identical CPU
// mirror above rather than the fixed point: seed a V blob and step both
// forward, so any error in the 9-point stencil, the ping-pong ordering,
// the interleave, or the barriers shows up as divergence.
TEST_CASE("Hal::Gpu::GrayScottKernel2D::RunSteps matches a CPU reference of the same update")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::ComputeContext &context = *pContext;

    const int32_t nx = 32;
    const int32_t ny = 48;
    const int32_t steps = 30;
    const float feed = 0.055f;
    const float kill = 0.062f;
    const float diffU = 0.16f;
    const float diffV = 0.08f;

    Hal::Gpu::GrayScottKernel2D kernel(context, nx, ny);

    std::vector<float> u(static_cast<size_t>(nx * ny), 1.0f);
    std::vector<float> v(static_cast<size_t>(nx * ny), 0.0f);

    // A small V-seeded square (with U depleted there), the standard
    // Gray-Scott perturbation that grows into a pattern.
    for (int32_t x = nx / 2 - 3; x < nx / 2 + 3; x++)
    {
        for (int32_t y = ny / 2 - 3; y < ny / 2 + 3; y++)
        {
            u[static_cast<size_t>(x * ny + y)] = 0.5f;
            v[static_cast<size_t>(x * ny + y)] = 0.25f;
        }
    }

    // Interleaved CPU state, ping-ponged the same way RunSteps chains.
    std::vector<float> cpuA(static_cast<size_t>(nx * ny) * 2);
    std::vector<float> cpuB(static_cast<size_t>(nx * ny) * 2);
    for (int32_t i = 0; i < nx * ny; i++)
    {
        cpuA[2 * static_cast<size_t>(i)] = u[static_cast<size_t>(i)];
        cpuA[2 * static_cast<size_t>(i) + 1] = v[static_cast<size_t>(i)];
    }

    std::vector<float> *pSrc = &cpuA;
    std::vector<float> *pDst = &cpuB;
    for (int32_t step = 0; step < steps; step++)
    {
        CpuGrayScottStep(*pSrc, *pDst, nx, ny, feed, kill, diffU, diffV);
        std::vector<float> *pTmp = pSrc;
        pSrc = pDst;
        pDst = pTmp;
    }

    kernel.RunSteps(u, v, steps, feed, kill, diffU, diffV);

    double maxDiff = 0.0;
    for (int32_t i = 0; i < nx * ny; i++)
    {
        double du = std::abs(static_cast<double>(u[static_cast<size_t>(i)]) - static_cast<double>((*pSrc)[2 * static_cast<size_t>(i)]));
        double dv = std::abs(static_cast<double>(v[static_cast<size_t>(i)]) - static_cast<double>((*pSrc)[2 * static_cast<size_t>(i) + 1]));
        maxDiff = std::max(maxDiff, std::max(du, dv));
    }
    CHECK(maxDiff < 1e-4);
}

TEST_CASE("Hal::Gpu::GrayScottKernel2D rejects U or V fields of the wrong length")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::GrayScottKernel2D kernel(*pContext, 4, 4);
    std::vector<float> rightSize(16, 0.0f);
    std::vector<float> wrongSize(15, 0.0f);
    CHECK_THROWS_AS(kernel.RunSteps(wrongSize, rightSize, 1, 0.05f, 0.06f, 0.2f, 0.1f), std::invalid_argument);
    CHECK_THROWS_AS(kernel.RunSteps(rightSize, wrongSize, 1, 0.05f, 0.06f, 0.2f, 0.1f), std::invalid_argument);
}
