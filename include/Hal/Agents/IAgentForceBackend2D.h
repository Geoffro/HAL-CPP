#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Strategy interface for the cell-bucket agent neighborhood force
    // accumulation, mirroring IDiffusionBackend2D for the explicit
    // diffusion stencil: model code holds a pointer to this and swaps a
    // CPU or GPU implementation without changing call sites. The
    // computation is the one Hal::Gpu::AgentForceKernel2D implements
    // (itself mirroring AgentGrid2D::GetAgentsRadApprox's bucket scan and
    // SphericalAgent2D::SumForces' interactionRad overload): for each
    // agent, scan the bounding box of grid cells around it and sum a
    // linear spring force (force = springConstant * ((rad_i + rad_j) -
    // dist)) over every other agent closer than interactionRad, into a
    // per-agent velocity delta.
    //
    // See Hal::CpuAgentForceBackend2D
    // (include/Hal/Agents/CpuAgentForceBackend2D.h) and
    // Hal::Gpu::GpuAgentForceBackend2D
    // (include/Hal/Gpu/GpuAgentForceBackend2D.h) for the implementations.
    class IAgentForceBackend2D
    {
    public:
        IAgentForceBackend2D();
        virtual ~IAgentForceBackend2D();

        IAgentForceBackend2D(const IAgentForceBackend2D &other) = delete;
        IAgentForceBackend2D &operator=(const IAgentForceBackend2D &other) = delete;

        // positions is a flat (x0, y0, x1, y1, ...) array; radii has one
        // entry per agent (the agent count is radii.size()).
        // velocityDeltas is resized to positions.size() and OVERWRITTEN
        // with each agent's summed (vx, vy) force contribution — matching
        // AgentForceKernel2D's per-dispatch semantics, not the diffusion
        // interface's accumulate-into semantics.
        virtual void AccumulateForces(const std::vector<double> &positions,
                                      const std::vector<double> &radii,
                                      std::vector<double>       &velocityDeltas,
                                      double                     interactionRad,
                                      double                     springConstant,
                                      int32_t                    xDim,
                                      int32_t                    yDim,
                                      bool                       wrapX,
                                      bool                       wrapY) = 0;
    };
}
