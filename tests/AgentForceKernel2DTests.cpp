#include <doctest/doctest.h>

#include "GpuTestSupport.h"

#include "Hal/Agents/SphericalAgent2D.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Gpu/AgentForceKernel2D.h"
#include "Hal/Gpu/ComputeContext.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

// Compares Hal::Gpu::AgentForceKernel2D with SphericalAgent2D::SumForces
// using the kernel's unclamped linear force, springConstant * touchDist.

class ForceTestGrid;

class ForceTestAgent : public Hal::SphericalAgent2D<ForceTestGrid>
{
public:
    explicit ForceTestAgent(ForceTestGrid *pGrid)
        : Hal::SphericalAgent2D<ForceTestGrid>(pGrid)
    {
    }

    ~ForceTestAgent()
    {
    }
};

class ForceTestGrid : public Hal::AgentGrid2D<ForceTestAgent, ForceTestGrid>
{
public:
    ForceTestGrid(int32_t  xDim,
                  int32_t  yDim,
                  bool     wrapX,
                  bool     wrapY)
        : Hal::AgentGrid2D<ForceTestAgent, ForceTestGrid>(xDim, yDim, wrapX, wrapY)
    {
    }

    ~ForceTestGrid()
    {
    }
};

TEST_CASE("Hal::Gpu::AgentForceKernel2D matches CPU SphericalAgent2D::SumForces with an unclamped linear spring force")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::ComputeContext &context = *pContext;

    const int32_t xDim = 20;
    const int32_t yDim = 20;
    const double interactionRad = 5.0;
    const double springConstant = 0.3;

    struct AgentSpec
    {
        double x;
        double y;
        double radius;
    };

    std::vector<AgentSpec> specs = {
        {9.0, 10.0, 2.0},
        {11.0, 10.0, 2.0},
        {1.0, 1.0, 1.5},
        {19.5, 1.0, 1.5},
        {5.0, 15.0, 1.0},
        {15.0, 5.0, 1.0},
    };

    ForceTestGrid grid(xDim, yDim, true, true);
    std::vector<ForceTestAgent *> agents;
    for (const AgentSpec &spec : specs)
    {
        ForceTestAgent &agent = grid.NewAgentPT(spec.x, spec.y);
        agent.DefaultInit(spec.radius);
        agents.push_back(&agent);
    }

    auto overlapFn = [&](double touchDist, ForceTestAgent & /*other*/)
    {
        return touchDist * springConstant;
    };

    std::vector<double> cpuVelX;
    std::vector<double> cpuVelY;
    for (ForceTestAgent *pAgent : agents)
    {
        pAgent->SumForces(interactionRad, overlapFn);
        cpuVelX.push_back(pAgent->XVel());
        cpuVelY.push_back(pAgent->YVel());
    }

    Hal::Gpu::AgentForceKernel2D kernel(context, xDim, yDim, static_cast<int32_t>(specs.size()));

    std::vector<float> positions;
    std::vector<float> radii;
    for (const AgentSpec &spec : specs)
    {
        positions.push_back(static_cast<float>(spec.x));
        positions.push_back(static_cast<float>(spec.y));
        radii.push_back(static_cast<float>(spec.radius));
    }

    kernel.UploadAgents(positions, radii, static_cast<int32_t>(specs.size()));
    kernel.Dispatch(static_cast<float>(interactionRad), static_cast<float>(springConstant), true, true);
    std::vector<float> velocityDeltas = kernel.DownloadVelocityDeltas();

    for (size_t i = 0; i < specs.size(); i++)
    {
        CAPTURE(i);
        CHECK(velocityDeltas[i * 2] == doctest::Approx(cpuVelX[i]).epsilon(1e-4));
        CHECK(velocityDeltas[i * 2 + 1] == doctest::Approx(cpuVelY[i]).epsilon(1e-4));
    }
}

TEST_CASE("Hal::Gpu::AgentForceKernel2D handles zero agents and rejects too many")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::AgentForceKernel2D kernel(*pContext, 8, 8, 4);
    std::vector<float> positions;
    std::vector<float> radii;
    kernel.UploadAgents(positions, radii, 0);
    kernel.Dispatch(2.0f, 1.0f, false, false);
    CHECK(kernel.DownloadVelocityDeltas().empty());

    positions.assign(10, 1.0f);
    radii.assign(5, 0.5f);
    CHECK_THROWS_AS(kernel.UploadAgents(positions, radii, 5), std::invalid_argument);
    CHECK_THROWS_AS(kernel.UploadAgents(positions, radii, -1), std::invalid_argument);
}
