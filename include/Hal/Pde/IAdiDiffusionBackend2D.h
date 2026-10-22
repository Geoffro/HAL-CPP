#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Strategy interface for one implicit (ADI) 2D diffusion step,
    // implemented on the CPU (Hal::CpuAdiDiffusionBackend2D) and the GPU
    // (Hal::Gpu::GpuAdiDiffusionBackend2D). Same flat-array layout as
    // IDiffusionBackend2D. Zero-flux or periodic boundaries.
    class IAdiDiffusionBackend2D
    {
    public:
        IAdiDiffusionBackend2D();
        virtual ~IAdiDiffusionBackend2D();

        IAdiDiffusionBackend2D(const IAdiDiffusionBackend2D &other) = delete;
        IAdiDiffusionBackend2D &operator=(const IAdiDiffusionBackend2D &other) = delete;

        // Adds one step's contribution into deltas (deltas[i] += ...); clear
        // deltas first if accumulation isn't wanted. Unconditionally
        // stable. Throws std::invalid_argument unless field and deltas hold
        // xDim*yDim values.
        virtual void DiffuseADI(const std::vector<double> &field,
                                std::vector<double>       &deltas,
                                double                     diffRate,
                                int32_t                    xDim,
                                int32_t                    yDim,
                                bool                       wrapX,
                                bool                       wrapY) = 0;
    };
}
