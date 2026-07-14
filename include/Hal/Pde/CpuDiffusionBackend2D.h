#pragma once

#include "Hal/Pde/IDiffusionBackend2D.h"

namespace Hal
{
    // CPU implementation of IDiffusionBackend2D: a thin wrapper over the
    // existing Hal::Internal::Diffusion2 stencil (see
    // include/Hal/Pde/PDEequations.h) — no new numerics, just the
    // interface adapter.
    class CpuDiffusionBackend2D : public IDiffusionBackend2D
    {
    public:
        CpuDiffusionBackend2D();
        ~CpuDiffusionBackend2D();

        void Diffuse(const std::vector<double> &field,
                     std::vector<double>       &deltas,
                     double                     diffRate,
                     int32_t                    xDim,
                     int32_t                    yDim,
                     bool                       wrapX,
                     bool                       wrapY) override;
    };
}
