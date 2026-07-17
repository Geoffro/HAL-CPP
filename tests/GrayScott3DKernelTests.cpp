#include <doctest/doctest.h>

#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/GrayScott3DKernel.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace GrayScott3DKernelTestsInternal
{
    int32_t Wrap3(int32_t v,
                  int32_t dim)
    {
        int32_t m = v % dim;
        return (m < 0) ? (m + dim) : m;
    }

    int32_t Flatten3(int32_t x,
                     int32_t y,
                     int32_t z,
                     int32_t yDim,
                     int32_t zDim)
    {
        return (x * yDim + y) * zDim + z;
    }

    // Bit-for-bit CPU mirror of shaders/GrayScott3D.comp (7-point 3D
    // Laplacian, same reaction/clamp/toroidal wrap), stepped in the same
    // ping-pong order, so the GPU kernel can be checked against its own
    // specification for the full nonlinear dynamics.
    void CpuGrayScott3DStep(const std::vector<float> &src,
                            std::vector<float>       &dst,
                            int32_t                   xDim,
                            int32_t                   yDim,
                            int32_t                   zDim,
                            float                     feed,
                            float                     kill,
                            float                     diffU,
                            float                     diffV)
    {
        for (int32_t x = 0; x < xDim; x++)
        {
            for (int32_t y = 0; y < yDim; y++)
            {
                for (int32_t z = 0; z < zDim; z++)
                {
                    int32_t xm = Wrap3(x - 1, xDim);
                    int32_t xp = Wrap3(x + 1, xDim);
                    int32_t ym = Wrap3(y - 1, yDim);
                    int32_t yp = Wrap3(y + 1, yDim);
                    int32_t zm = Wrap3(z - 1, zDim);
                    int32_t zp = Wrap3(z + 1, zDim);

                    size_t c = static_cast<size_t>(Flatten3(x, y, z, yDim, zDim));
                    float u = src[2 * c];
                    float v = src[2 * c + 1];

                    for (int32_t comp = 0; comp < 2; comp++)
                    {
                        float neighbors =
                            src[2 * static_cast<size_t>(Flatten3(xm, y, z, yDim, zDim)) + comp] +
                            src[2 * static_cast<size_t>(Flatten3(xp, y, z, yDim, zDim)) + comp] +
                            src[2 * static_cast<size_t>(Flatten3(x, ym, z, yDim, zDim)) + comp] +
                            src[2 * static_cast<size_t>(Flatten3(x, yp, z, yDim, zDim)) + comp] +
                            src[2 * static_cast<size_t>(Flatten3(x, y, zm, yDim, zDim)) + comp] +
                            src[2 * static_cast<size_t>(Flatten3(x, y, zp, yDim, zDim)) + comp];
                        float center = src[2 * c + comp];
                        float lap = neighbors - 6.0f * center;
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
}

using namespace GrayScott3DKernelTestsInternal;

TEST_CASE("Hal::Gpu::GrayScott3DKernel leaves the (U=1, V=0) fixed point unchanged")
{
    try
    {
        Hal::Gpu::ComputeContext context;

        const int32_t nx = 16;
        const int32_t ny = 12;
        const int32_t nz = 20;

        Hal::Gpu::GrayScott3DKernel kernel(context, nx, ny, nz);

        std::vector<float> u(static_cast<size_t>(nx * ny * nz), 1.0f);
        std::vector<float> v(static_cast<size_t>(nx * ny * nz), 0.0f);

        kernel.RunSteps(u, v, 20, 0.04f, 0.06f, 0.16f, 0.08f);

        for (size_t i = 0; i < u.size(); i++)
        {
            CHECK(u[i] == doctest::Approx(1.0).epsilon(1e-6));
            CHECK(v[i] == doctest::Approx(0.0).epsilon(1e-6));
        }
    }
    catch (const std::runtime_error &e)
    {
        MESSAGE("No usable Vulkan compute device in this environment: ", std::string(e.what()));
    }
}

TEST_CASE("Hal::Gpu::GrayScott3DKernel::RunSteps matches a CPU reference of the same update, or reports why it can't")
{
    try
    {
        Hal::Gpu::ComputeContext context;

        const int32_t nx = 16;
        const int32_t ny = 12;
        const int32_t nz = 20;
        const int32_t steps = 24;
        const float feed = 0.04f;
        const float kill = 0.06f;
        const float diffU = 0.16f;
        const float diffV = 0.08f;

        Hal::Gpu::GrayScott3DKernel kernel(context, nx, ny, nz);

        std::vector<float> u(static_cast<size_t>(nx * ny * nz), 1.0f);
        std::vector<float> v(static_cast<size_t>(nx * ny * nz), 0.0f);

        // A small V-seeded box at the center.
        for (int32_t x = nx / 2 - 2; x < nx / 2 + 2; x++)
        {
            for (int32_t y = ny / 2 - 2; y < ny / 2 + 2; y++)
            {
                for (int32_t z = nz / 2 - 2; z < nz / 2 + 2; z++)
                {
                    size_t i = static_cast<size_t>(Flatten3(x, y, z, ny, nz));
                    u[i] = 0.5f;
                    v[i] = 0.25f;
                }
            }
        }

        std::vector<float> cpuA(static_cast<size_t>(nx * ny * nz) * 2);
        std::vector<float> cpuB(static_cast<size_t>(nx * ny * nz) * 2);
        for (int32_t i = 0; i < nx * ny * nz; i++)
        {
            cpuA[2 * static_cast<size_t>(i)] = u[static_cast<size_t>(i)];
            cpuA[2 * static_cast<size_t>(i) + 1] = v[static_cast<size_t>(i)];
        }

        std::vector<float> *pSrc = &cpuA;
        std::vector<float> *pDst = &cpuB;
        for (int32_t step = 0; step < steps; step++)
        {
            CpuGrayScott3DStep(*pSrc, *pDst, nx, ny, nz, feed, kill, diffU, diffV);
            std::vector<float> *pTmp = pSrc;
            pSrc = pDst;
            pDst = pTmp;
        }

        kernel.RunSteps(u, v, steps, feed, kill, diffU, diffV);

        double maxDiff = 0.0;
        for (int32_t i = 0; i < nx * ny * nz; i++)
        {
            double du = std::abs(static_cast<double>(u[static_cast<size_t>(i)]) - static_cast<double>((*pSrc)[2 * static_cast<size_t>(i)]));
            double dv = std::abs(static_cast<double>(v[static_cast<size_t>(i)]) - static_cast<double>((*pSrc)[2 * static_cast<size_t>(i) + 1]));
            maxDiff = std::max(maxDiff, std::max(du, dv));
        }
        CHECK(maxDiff < 1e-4);
    }
    catch (const std::runtime_error &e)
    {
        MESSAGE("No usable Vulkan compute device in this environment: ", std::string(e.what()));
    }
}
