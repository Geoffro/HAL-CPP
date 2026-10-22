#include <doctest/doctest.h>

#include "GpuTestSupport.h"

#include "Hal/Agents/CpuAgentForceBackend2D.h"
#include "Hal/Agents/IAgentForceBackend2D.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/GpuAgentForceBackend2D.h"
#include "Hal/Rand.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace AgentForceBackendTestHelpers
{
    const int32_t g_gridDim = 16;
    const int32_t g_agentCount = 60;
    const double  g_interactionRad = 3.0;
    const double  g_springConstant = 0.4;

    // Deterministic scene shared by every backend check.
    void BuildScene(std::vector<double> &positions,
                    std::vector<double> &radii)
    {
        Hal::Rand rand(1234);
        positions.clear();
        radii.clear();
        for (int32_t i = 0; i < g_agentCount; i++)
        {
            positions.push_back(rand.Double(static_cast<double>(g_gridDim)));
            positions.push_back(rand.Double(static_cast<double>(g_gridDim)));
            radii.push_back(0.4 + rand.Double(0.4));
        }
    }

    double WrapDisp(double  a,
                    double  b,
                    int32_t dim)
    {
        double d = b - a;
        if (std::fabs(d) > (dim * 0.5))
        {
            d = (a > b) ? (d + dim) : (d - dim);
        }
        return d;
    }

    // Independent ground truth: a brute-force all-pairs loop with the
    // same linear spring force law, no cell buckets at all — so a bug in
    // the bucket structure shared by the CPU and GPU backends could not
    // hide behind their agreement.
    std::vector<double> BruteForceDeltas(const std::vector<double> &positions,
                                         const std::vector<double> &radii)
    {
        std::vector<double> expected(positions.size(), 0.0);
        for (int32_t i = 0; i < g_agentCount; i++)
        {
            for (int32_t j = 0; j < g_agentCount; j++)
            {
                if (j == i)
                {
                    continue;
                }
                double xComp = WrapDisp(positions[static_cast<size_t>(i) * 2], positions[static_cast<size_t>(j) * 2], g_gridDim);
                double yComp = WrapDisp(positions[static_cast<size_t>(i) * 2 + 1], positions[static_cast<size_t>(j) * 2 + 1], g_gridDim);
                double dist = std::sqrt((xComp * xComp) + (yComp * yComp));
                if ((dist < g_interactionRad) && (dist > 0.0))
                {
                    double force = g_springConstant * ((radii[static_cast<size_t>(i)] + radii[static_cast<size_t>(j)]) - dist);
                    expected[static_cast<size_t>(i) * 2] -= (xComp / dist) * force;
                    expected[static_cast<size_t>(i) * 2 + 1] -= (yComp / dist) * force;
                }
            }
        }
        return expected;
    }

    // The single call site both backends run through unmodified — the
    // point of the strategy interface.
    void RunForceCheck(Hal::IAgentForceBackend2D &backend,
                       double                     tolerance)
    {
        std::vector<double> positions;
        std::vector<double> radii;
        BuildScene(positions, radii);
        std::vector<double> expected = BruteForceDeltas(positions, radii);

        std::vector<double> actual;
        backend.AccumulateForces(positions, radii, actual, g_interactionRad, g_springConstant, g_gridDim, g_gridDim, true, true);

        REQUIRE(actual.size() == expected.size());
        for (size_t k = 0; k < expected.size(); k++)
        {
            CAPTURE(k);
            CHECK(actual[k] == doctest::Approx(expected[k]).epsilon(tolerance).scale(1.0));
        }
    }
}

TEST_CASE("Hal::CpuAgentForceBackend2D matches an independent brute-force all-pairs force sum")
{
    Hal::CpuAgentForceBackend2D backend;
    AgentForceBackendTestHelpers::RunForceCheck(backend, 1e-12);
}

TEST_CASE("Hal::Gpu::GpuAgentForceBackend2D matches the same brute-force ground truth through the same call site")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::ComputeContext &context = *pContext;
    Hal::Gpu::GpuAgentForceBackend2D backend(context,
                                             AgentForceBackendTestHelpers::g_gridDim,
                                             AgentForceBackendTestHelpers::g_gridDim,
                                             AgentForceBackendTestHelpers::g_agentCount);
    AgentForceBackendTestHelpers::RunForceCheck(backend, 1e-3);

    // Dimension and array-size misuse throws rather than silently
    // misrendering.
    std::vector<double> positions;
    std::vector<double> radii;
    AgentForceBackendTestHelpers::BuildScene(positions, radii);
    std::vector<double> deltas;
    CHECK_THROWS_AS(backend.AccumulateForces(positions, radii, deltas, 1.0, 1.0, 8, 8, true, true), std::invalid_argument);

    positions.pop_back();
    CHECK_THROWS_AS(backend.AccumulateForces(positions,
                                             radii,
                                             deltas,
                                             1.0,
                                             1.0,
                                             AgentForceBackendTestHelpers::g_gridDim,
                                             AgentForceBackendTestHelpers::g_gridDim,
                                             true,
                                             true),
                    std::invalid_argument);
}

TEST_CASE("Hal::CpuAgentForceBackend2D rejects positions that don't hold two values per radius")
{
    Hal::CpuAgentForceBackend2D backend;
    std::vector<double> positions = {1.0, 1.0, 2.0};
    std::vector<double> radii = {0.5, 0.5};
    std::vector<double> deltas;
    CHECK_THROWS_AS(backend.AccumulateForces(positions, radii, deltas, 2.0, 1.0, 8, 8, false, false), std::invalid_argument);
}

TEST_CASE("Hal::CpuAgentForceBackend2D skips coincident agents instead of producing NaN")
{
    Hal::CpuAgentForceBackend2D backend;
    std::vector<double> positions = {4.0, 4.0, 4.0, 4.0};
    std::vector<double> radii = {0.5, 0.5};
    std::vector<double> deltas;
    backend.AccumulateForces(positions, radii, deltas, 2.0, 1.0, 8, 8, false, false);
    REQUIRE(deltas.size() == 4);
    for (double v : deltas)
    {
        CHECK(v == 0.0);
    }
}
