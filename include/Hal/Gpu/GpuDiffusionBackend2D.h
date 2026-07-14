#pragma once

#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/DiffusionKernel2D.h"
#include "Hal/Pde/IDiffusionBackend2D.h"

namespace Hal
{
    namespace Gpu
    {
        // GPU implementation of IDiffusionBackend2D, wrapping
        // DiffusionKernel2D behind the same interface
        // CpuDiffusionBackend2D implements — this is the
        // docs/PHASED_PORT.md "IComputeBackend" milestone: model code can
        // hold an Hal::IDiffusionBackend2D* and swap CPU/GPU without
        // changing call sites.
        //
        // Fixed-size: the wrapped DiffusionKernel2D's buffers are sized
        // once at construction (matching xDim/yDim given here), so
        // Diffuse() throws std::invalid_argument if called with different
        // dimensions rather than silently resizing — same "correctness
        // first" stance as DiffusionKernel2D itself.
        //
        // Diffuse() converts field/deltas between the interface's double
        // arrays and the kernel's float SSBOs, and reads deltas back as
        // deltas[i] += result (matching the interface's accumulate
        // semantics) rather than overwriting — the conversion and the
        // extra host round-trip are the acceptable cost of reusing one
        // interface across a double-precision CPU path and a float GPU
        // one; a same-precision fast path is a legitimate follow-up, not
        // a correctness requirement now.
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
