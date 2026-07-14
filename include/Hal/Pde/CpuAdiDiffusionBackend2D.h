#pragma once

#include "Hal/Pde/IAdiDiffusionBackend2D.h"
#include "Hal/Pde/TdmaSolver.h"

#include <vector>

namespace Hal
{
    // CPU implementation of IAdiDiffusionBackend2D: a thin wrapper over
    // the existing Hal::Internal::Diffusion2DADI directional-sweep solver
    // (see include/Hal/Pde/ADIequations.h) — no new numerics, just the
    // interface adapter. Owns the TDMA solver and the intermediate-sweep
    // scratch buffer PDEGrid2D::DiffusionADI otherwise carries (the
    // scratch is resized on first use / dimension change).
    class CpuAdiDiffusionBackend2D : public IAdiDiffusionBackend2D
    {
    public:
        CpuAdiDiffusionBackend2D();
        ~CpuAdiDiffusionBackend2D();

        void DiffuseADI(const std::vector<double> &field,
                        std::vector<double>       &deltas,
                        double                     diffRate,
                        int32_t                    xDim,
                        int32_t                    yDim,
                        bool                       wrapX,
                        bool                       wrapY) override;

    private:
        TdmaSolver          m_tdma;
        std::vector<double> m_scratch;
    };
}
