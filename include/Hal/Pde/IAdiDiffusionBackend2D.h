#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Strategy interface for the implicit (ADI / Crank-Nicolson) 2D
    // diffusion solve, mirroring IDiffusionBackend2D for the explicit
    // stencil: model code holds a pointer to this and swaps
    // implementations without changing call sites. Operates on flat,
    // caller-owned field/delta arrays (matching
    // Hal::Internal::Diffusion2DADI's raw-pointer signature) so one
    // interface covers a plain array and a PDEGrid2D's backing storage.
    //
    // Currently one implementation exists — Hal::CpuAdiDiffusionBackend2D
    // (include/Hal/Pde/CpuAdiDiffusionBackend2D.h). A GPU implementation
    // (tridiagonal solve via cyclic reduction / PCR) is the deferred
    // Phase 9 GPU-ADI item in docs/PHASED_PORT.md; it will slot in behind
    // this same interface.
    class IAdiDiffusionBackend2D
    {
    public:
        IAdiDiffusionBackend2D();
        virtual ~IAdiDiffusionBackend2D();

        IAdiDiffusionBackend2D(const IAdiDiffusionBackend2D &other) = delete;
        IAdiDiffusionBackend2D &operator=(const IAdiDiffusionBackend2D &other) = delete;

        // Adds one ADI step's contribution into deltas (deltas[i] += ...),
        // matching Hal::Internal::Diffusion2DADI's accumulate semantics —
        // callers are responsible for clearing deltas first if that's not
        // wanted. Unconditionally stable: no CFL cap on diffRate.
        virtual void DiffuseADI(const std::vector<double> &field,
                                std::vector<double>       &deltas,
                                double                     diffRate,
                                int32_t                    xDim,
                                int32_t                    yDim,
                                bool                       wrapX,
                                bool                       wrapY) = 0;
    };
}
