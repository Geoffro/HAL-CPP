#pragma once

#include "Hal/Agents/IAgentForceBackend2D.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    // CPU IAgentForceBackend2D. Uses the same four passes as the GPU kernel
    // (count agents per cell, prefix-sum into offsets, scatter indices,
    // per-agent scan and force sum) so the two backends agree to float
    // precision. Scratch buffers are reused across calls.
    class CpuAgentForceBackend2D : public IAgentForceBackend2D
    {
    public:
        CpuAgentForceBackend2D();
        ~CpuAgentForceBackend2D();

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
        std::vector<int32_t> m_cellOffsets;
        std::vector<int32_t> m_cellCursors;
        std::vector<int32_t> m_bucketAgents;
    };
}
