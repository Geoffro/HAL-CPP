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
        // GPU IAdiDiffusionBackend2D over AdiKernel2D. Sized at construction;
        // DiffuseADI throws std::invalid_argument for other dimensions. Converts
        // between double and float each call; to step many times, use
        // AdiKernel2D::RunSteps directly.
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
