#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Strategy interface so model code can pick a CPU or GPU explicit-
    // diffusion implementation without changing call sites (the
    // IComputeBackend pattern). Operates on
    // flat, caller-owned field/delta arrays (matching
    // Hal::Internal::Diffusion2's raw-pointer signature and
    // Diffusion2D.comp's SSBO layout) rather than a PDEGrid2D directly, so
    // a single interface covers both a plain array and a PDEGrid2D's own
    // backing storage.
    //
    // See Hal::CpuDiffusionBackend2D (include/Hal/Pde/CpuDiffusionBackend2D.h)
    // and Hal::Gpu::GpuDiffusionBackend2D
    // (include/Hal/Gpu/GpuDiffusionBackend2D.h) for the two implementations.
    class IDiffusionBackend2D
    {
    public:
        IDiffusionBackend2D();
        virtual ~IDiffusionBackend2D();

        IDiffusionBackend2D(const IDiffusionBackend2D &other) = delete;
        IDiffusionBackend2D &operator=(const IDiffusionBackend2D &other) = delete;

        // Adds one explicit-diffusion step's contribution into deltas
        // (deltas[i] += ...), matching Hal::Internal::Diffusion2 and
        // Diffusion2D.comp's accumulate-not-overwrite semantics — callers
        // are responsible for clearing deltas first if that's not wanted.
        virtual void Diffuse(const std::vector<double> &field,
                             std::vector<double>       &deltas,
                             double                     diffRate,
                             int32_t                    xDim,
                             int32_t                    yDim,
                             bool                       wrapX,
                             bool                       wrapY) = 0;
    };
}
