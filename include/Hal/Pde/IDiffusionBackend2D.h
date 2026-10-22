#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Strategy interface for one explicit 2D diffusion step, implemented on
    // the CPU (Hal::CpuDiffusionBackend2D) and the GPU
    // (Hal::Gpu::GpuDiffusionBackend2D) so model code is backend-agnostic.
    // Operates on flat, caller-owned field/delta arrays (index x*yDim + y),
    // e.g. a PDEGrid2D's own storage.
    class IDiffusionBackend2D
    {
    public:
        IDiffusionBackend2D();
        virtual ~IDiffusionBackend2D();

        IDiffusionBackend2D(const IDiffusionBackend2D &other) = delete;
        IDiffusionBackend2D &operator=(const IDiffusionBackend2D &other) = delete;

        // Adds one step's contribution into deltas (deltas[i] += ...); clear
        // deltas first if accumulation isn't wanted. Throws
        // std::invalid_argument unless field and deltas hold xDim*yDim
        // values.
        virtual void Diffuse(const std::vector<double> &field,
                             std::vector<double>       &deltas,
                             double                     diffRate,
                             int32_t                    xDim,
                             int32_t                    yDim,
                             bool                       wrapX,
                             bool                       wrapY) = 0;
    };

    namespace Internal
    {
        // Throws std::invalid_argument (after logging) unless field and
        // deltas both hold xDim*yDim values. Shared by the diffusion
        // backends.
        void CheckFieldSizes(const std::vector<double> &field,
                             const std::vector<double> &deltas,
                             int32_t                    xDim,
                             int32_t                    yDim,
                             const char                *pCaller);
    }
}
