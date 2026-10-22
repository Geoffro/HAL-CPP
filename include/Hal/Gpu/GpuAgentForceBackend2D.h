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
        // GPU IAgentForceBackend2D over AgentForceKernel2D. Sized at
        // construction (grid and agent capacity); AccumulateForces throws
        // std::invalid_argument for other dimensions or more agents. Converts
        // between double and float each call.
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
