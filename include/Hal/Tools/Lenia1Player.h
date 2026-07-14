#pragma once

#include "Hal/Grids/Grid2D.h"
#include "Hal/Tools/FftGrid.h"

#include <cstdint>
#include <functional>

namespace Hal
{
    // Single-channel Lenia, a continuous cellular automaton, ported from
    // Tools/Lenia/Lenia1Player.java. Each step convolves the state field A
    // with a fixed radial kernel K (via the FFT — see Hal::FftGrid), maps
    // the resulting per-cell "potential" through a growth function G, and
    // integrates: A <- clamp(A + G(K * A) * deltaT, 0, 1).
    //
    // The kernel and growth functions are std::function<double(double)>:
    // they are stored, swappable at runtime (SetKernelFunction /
    // SetGrowthFunction, like the Java setters), and typically capture
    // parameters (kernel radius, growth mean/width). This is the
    // stored-configurable-callback case, distinct from the hot per-call
    // PDE boundary callbacks that docs/DESIGN.md's callback-style rule
    // targets — the growth call, though per-cell, is a simple scalar map
    // and correctness/runtime-flexibility matter more here than shaving
    // the type-erasure indirection.
    //
    // KernelFn maps a distance-from-the-focal-cell to a kernel weight;
    // GrowthFn maps a convolved potential to a growth rate.
    class Lenia1Player
    {
    public:
        // sideLenExp sets the grid side length to 2^sideLenExp (kept a
        // power of two for the FFT). deltaT is the integration step (1/T).
        Lenia1Player(int32_t                        sideLenExp,
                     double                         deltaT,
                     std::function<double(double)>  kernelFn,
                     std::function<double(double)>  growthFn);
        ~Lenia1Player();

        Lenia1Player(const Lenia1Player &other) = delete;
        Lenia1Player &operator=(const Lenia1Player &other) = delete;

        void SetKernelFunction(std::function<double(double)> kernelFn);
        void SetGrowthFunction(std::function<double(double)> growthFn);

        // Recomputes the (normalized, FFT-transformed) kernel from the
        // current kernel function. Called automatically by the constructor
        // and SetKernelFunction; call it directly if the kernel function's
        // captured parameters change in place.
        void RecalcKernel();

        // Advances the automaton one step (convolve, grow, integrate,
        // clamp), incrementing the tick.
        void Update();

        int32_t Xdim() const;
        int32_t Ydim() const;
        int32_t Length() const;
        int32_t GetTick() const;
        void ResetTick();

        double Get(int32_t i) const;
        double Get(int32_t x,
                   int32_t y) const;
        void Set(int32_t i,
                 double  val);
        void Set(int32_t x,
                 int32_t y,
                 double  val);

        const Grid2D<double> &GetField() const;

        // The convolved potential (K * A) from the most recent Update, i.e.
        // the value passed to the growth function at each cell.
        double GetConvolvedFieldVal(int32_t i) const;
        double GetConvolvedFieldVal(int32_t x,
                                    int32_t y) const;

        // The (un-normalized) kernel weight at a cell, for visualization.
        double GetKernelVal(int32_t i) const;
        double GetKernelVal(int32_t x,
                            int32_t y) const;

    private:
        int32_t m_xDim;
        int32_t m_yDim;
        int32_t m_length;
        double m_deltaT;
        int32_t m_tick;
        double m_kSum;
        std::function<double(double)> m_kernelFn;
        std::function<double(double)> m_growthFn;
        Grid2D<double> m_field;
        Grid2D<double> m_kernelField;
        FftGrid m_fftField;
        FftGrid m_fftKernel;
    };
}
