#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Strategy interface for cell-bucket agent force accumulation,
    // implemented on the CPU (Hal::CpuAgentForceBackend2D) and the GPU
    // (Hal::Gpu::GpuAgentForceBackend2D). For each agent, scans the grid
    // cells around it and sums a linear spring force,
    // springConstant * ((rad_i + rad_j) - dist), over every other agent
    // closer than interactionRad (and not at the same position), as in
    // SphericalAgent2D::SumForces.
    class IAgentForceBackend2D
    {
    public:
        IAgentForceBackend2D();
        virtual ~IAgentForceBackend2D();

        IAgentForceBackend2D(const IAgentForceBackend2D &other) = delete;
        IAgentForceBackend2D &operator=(const IAgentForceBackend2D &other) = delete;

        // positions is a flat (x0, y0, x1, y1, ...) array; radii has one
        // entry per agent. velocityDeltas is resized to positions.size() and
        // overwritten (not accumulated) with each agent's (vx, vy). Throws
        // std::invalid_argument unless positions.size() == 2 * radii.size().
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
