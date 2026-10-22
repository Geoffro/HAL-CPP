#include "Hal/Pde/CpuDiffusionBackend2D.h"

#include "Hal/Pde/PDEequations.h"

namespace Hal
{
    CpuDiffusionBackend2D::CpuDiffusionBackend2D()
    {
    }

    CpuDiffusionBackend2D::~CpuDiffusionBackend2D()
    {
    }

    void CpuDiffusionBackend2D::Diffuse(const std::vector<double> &field,
                                        std::vector<double>       &deltas,
                                        double                     diffRate,
                                        int32_t                    xDim,
                                        int32_t                    yDim,
                                        bool                       wrapX,
                                        bool                       wrapY)
    {
        Internal::CheckFieldSizes(field, deltas, xDim, yDim, "CpuDiffusionBackend2D::Diffuse");
        Internal::Diffusion2(field.data(), deltas.data(), diffRate, xDim, yDim, wrapX, wrapY);
    }
}
