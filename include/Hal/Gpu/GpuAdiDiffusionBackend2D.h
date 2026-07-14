#pragma once

#include "Hal/Gpu/AdiKernel2D.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Pde/IAdiDiffusionBackend2D.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    namespace Gpu
    {
        // GPU implementation of IAdiDiffusionBackend2D, wrapping
        // AdiKernel2D behind the same interface CpuAdiDiffusionBackend2D
        // implements — the last compute path to gain a GPU backend
        // (docs/PHASED_PORT.md's Phase 9 GPU-ADI item).
        //
        // Fixed-size, throw-on-misuse, and double<->float converting,
        // same stance as GpuDiffusionBackend2D. Each DiffuseADI call runs
        // one ADI step through AdiKernel2D::RunSteps and accumulates
        // (result - field) into deltas, matching the interface's
        // contract; a model stepping many times should prefer the
        // kernel's RunSteps directly, for the same GPU-residency reason
        // as explicit diffusion (see DESIGN.md).
        class GpuAdiDiffusionBackend2D : public IAdiDiffusionBackend2D
        {
        public:
            GpuAdiDiffusionBackend2D(ComputeContext &context,
                                     int32_t         xDim,
                                     int32_t         yDim);
            ~GpuAdiDiffusionBackend2D();

            void DiffuseADI(const std::vector<double> &field,
                            std::vector<double>       &deltas,
                            double                     diffRate,
                            int32_t                    xDim,
                            int32_t                    yDim,
                            bool                       wrapX,
                            bool                       wrapY) override;

        private:
            int32_t     m_xDim;
            int32_t     m_yDim;
            AdiKernel2D m_kernel;
        };
    }
}
