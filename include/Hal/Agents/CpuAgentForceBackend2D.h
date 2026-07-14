#pragma once

#include "Hal/Agents/IAgentForceBackend2D.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    // CPU implementation of IAgentForceBackend2D. Deliberately mirrors
    // the GPU kernel's four passes rather than reusing the
    // SphericalAgent2D machinery (which needs a whole agent grid/pool):
    // count agents per cell, prefix-sum the counts into offsets, scatter
    // agent indices into their cells' slot ranges, then per agent scan
    // the bounding box of cells around it and sum the linear spring
    // force — so the two backends walk the same buckets in the same
    // structure and can be compared to float precision. Scratch buffers
    // are class members reused across calls.
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
