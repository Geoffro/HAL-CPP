#include "Hal/Gpu/GpuAgentForceBackend2D.h"

#include <stdexcept>

namespace Hal
{
    namespace Gpu
    {
        GpuAgentForceBackend2D::GpuAgentForceBackend2D(ComputeContext &context,
                                                       int32_t         xDim,
                                                       int32_t         yDim,
                                                       int32_t         maxAgents)
            : m_xDim(xDim),
              m_yDim(yDim),
              m_maxAgents(maxAgents),
              m_kernel(context, xDim, yDim, maxAgents)
        {
        }

        GpuAgentForceBackend2D::~GpuAgentForceBackend2D()
        {
        }

        void GpuAgentForceBackend2D::AccumulateForces(const std::vector<double> &positions,
                                                      const std::vector<double> &radii,
                                                      std::vector<double>       &velocityDeltas,
                                                      double                     interactionRad,
                                                      double                     springConstant,
                                                      int32_t                    xDim,
                                                      int32_t                    yDim,
                                                      bool                       wrapX,
                                                      bool                       wrapY)
        {
            if ((xDim != m_xDim) || (yDim != m_yDim))
            {
                throw std::invalid_argument("GpuAgentForceBackend2D::AccumulateForces called with dimensions differing from construction");
            }
            int32_t numAgents = static_cast<int32_t>(radii.size());
            if (numAgents > m_maxAgents)
            {
                throw std::invalid_argument("GpuAgentForceBackend2D::AccumulateForces called with more agents than the construction-time capacity");
            }

            std::vector<float> floatPositions(positions.size());
            for (size_t k = 0; k < positions.size(); k++)
            {
                floatPositions[k] = static_cast<float>(positions[k]);
            }
            std::vector<float> floatRadii(radii.size());
            for (size_t k = 0; k < radii.size(); k++)
            {
                floatRadii[k] = static_cast<float>(radii[k]);
            }

            m_kernel.UploadAgents(floatPositions, floatRadii, numAgents);
            m_kernel.Dispatch(static_cast<float>(interactionRad), static_cast<float>(springConstant), wrapX, wrapY);
            std::vector<float> floatDeltas = m_kernel.DownloadVelocityDeltas();

            velocityDeltas.assign(floatDeltas.size(), 0.0);
            for (size_t k = 0; k < floatDeltas.size(); k++)
            {
                velocityDeltas[k] = static_cast<double>(floatDeltas[k]);
            }
        }
    }
}
