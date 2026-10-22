#pragma once

#include "Hal/Pde/IAdiDiffusionBackend2D.h"
#include "Hal/Pde/IDiffusionBackend2D.h"
#include "Hal/Pde/TdmaSolver.h"

#include <vector>

namespace Hal
{
    // CPU IAdiDiffusionBackend2D over Hal::Internal::Diffusion2DADI. Owns
    // the TDMA solver and the between-sweep scratch buffer.
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
