#pragma once

#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/DiffusionKernel2D.h"
#include "Hal/Pde/IDiffusionBackend2D.h"

namespace Hal
{
    namespace Gpu
    {
        // GPU IDiffusionBackend2D over DiffusionKernel2D. Sized at construction;
        // Diffuse throws std::invalid_argument for other dimensions. Converts
        // between double and float each call; to step many times, use
        // DiffusionKernel2D::RunSteps directly.
        class GpuDiffusionBackend2D : public IDiffusionBackend2D
        {
        public:
            GpuDiffusionBackend2D(ComputeContext &context,
                                  int32_t         xDim,
                                  int32_t         yDim);
            ~GpuDiffusionBackend2D();

            void Diffuse(const std::vector<double> &field,
                         std::vector<double>       &deltas,
                         double                     diffRate,
                         int32_t                    xDim,
                         int32_t                    yDim,
                         bool                       wrapX,
                         bool                       wrapY) override;

        private:
            int32_t m_xDim;
            int32_t m_yDim;
            DiffusionKernel2D m_kernel;
        };
    }
}
