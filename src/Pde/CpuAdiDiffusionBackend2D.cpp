#include "Hal/Pde/CpuAdiDiffusionBackend2D.h"

#include "Hal/Pde/ADIequations.h"

namespace Hal
{
    // The TDMA scratch starts empty — TdmaSolver grows it to each solve's
    // length on demand, so no dimensions are needed up front.
    CpuAdiDiffusionBackend2D::CpuAdiDiffusionBackend2D()
        : m_tdma(0),
          m_scratch()
    {
    }

    CpuAdiDiffusionBackend2D::~CpuAdiDiffusionBackend2D()
    {
    }

    void CpuAdiDiffusionBackend2D::DiffuseADI(const std::vector<double> &field,
                                              std::vector<double>       &deltas,
                                              double                     diffRate,
                                              int32_t                    xDim,
                                              int32_t                    yDim,
                                              bool                       wrapX,
                                              bool                       wrapY)
    {
        size_t length = static_cast<size_t>(xDim) * static_cast<size_t>(yDim);
        if (m_scratch.size() != length)
        {
            m_scratch.assign(length, 0.0);
        }
        Internal::Diffusion2DADI(field.data(), m_scratch.data(), deltas.data(), diffRate, xDim, yDim, wrapX, wrapY, m_tdma);
    }
}
