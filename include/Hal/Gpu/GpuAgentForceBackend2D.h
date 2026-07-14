#pragma once

#include "Hal/Agents/IAgentForceBackend2D.h"
#include "Hal/Gpu/AgentForceKernel2D.h"
#include "Hal/Gpu/ComputeContext.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    namespace Gpu
    {
        // GPU implementation of IAgentForceBackend2D, wrapping
        // AgentForceKernel2D behind the same interface
        // CpuAgentForceBackend2D implements — the agent-force analogue of
        // GpuDiffusionBackend2D, extending the IComputeBackend strategy
        // pattern to the second compute path (docs/PHASED_PORT.md
        // Phase 9).
        //
        // Fixed-size: the wrapped kernel's buffers are sized once at
        // construction (grid dimensions and agent capacity), so
        // AccumulateForces() throws std::invalid_argument if called with
        // different dimensions or more agents than maxAgents — same
        // stance as GpuDiffusionBackend2D.
        //
        // Converts between the interface's double arrays and the kernel's
        // float SSBOs; the conversion cost is the accepted price of one
        // interface spanning a double-precision CPU path and a float GPU
        // one (see GpuDiffusionBackend2D's header).
        class GpuAgentForceBackend2D : public IAgentForceBackend2D
        {
        public:
            GpuAgentForceBackend2D(ComputeContext &context,
                                   int32_t         xDim,
                                   int32_t         yDim,
                                   int32_t         maxAgents);
            ~GpuAgentForceBackend2D();

            void AccumulateForces(const std::vector<double> &positions,
                                  const std::vector<double> &radii,
                                  std::vector<double>       &velocityDeltas,
                                  double                     interactionRad,
                                  double                     springConstant,
                                  int32_t                    xDim,
                                  int32_t                    yDim,
                                  bool                       wrapX,
                                  bool                       wrapY) override;

        private:
            int32_t            m_xDim;
            int32_t            m_yDim;
            int32_t            m_maxAgents;
            AgentForceKernel2D m_kernel;
        };
    }
}
