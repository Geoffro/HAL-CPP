#pragma once

#include "Hal/Grids/Grid2D.h"
#include "Hal/Tools/FftGrid.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace Hal
{
    // One labeled kernel snapshot held by LeniaNPlayer::StoreKernel: the
    // kernel function plus every derived per-pair kernel artifact
    // (spatial fields, normalization sums, FFT-transformed kernels), so
    // LoadKernel restores the exact convolution state without a
    // RecalcKernels. Deep copies, unlike Java's KernelInfo, which stashed
    // references and relied on RecalcKernels reallocating (safe only
    // under a garbage collector).
    struct LeniaKernelSnapshot
    {
        std::function<double(int32_t, int32_t, double)> kernelFn;
        std::vector<Grid2D<double>>                     kernelFields;
        std::vector<double>                             kSums;
        std::vector<FftGrid>                            fftKernels;
    };

    // Multi-channel Lenia, ported from Tools/Lenia/LeniaNPlayer.java —
    // several coupled continuous-CA "player" fields A_0..A_{n-1}. For each
    // ordered channel pair (i, j) there is a kernel K_ij describing how
    // source channel j influences target channel i. Each step convolves
    // every source field A_j with every K_ij (via Hal::FftGrid), then
    // updates each channel i from the growth function applied to that
    // channel's full set of convolutions:
    //   A_i <- clamp(A_i + G(i, [K_i0*A_0, ..., K_i(n-1)*A_(n-1)]) * dt, 0, 1)
    //
    // KernelFn(targetChannel, sourceChannel, distanceFromFocalCell) -> weight.
    // GrowthFn(channel, convolvedValues) -> growth rate, where
    // convolvedValues holds the n per-source convolutions (K_ij*A_j) at one
    // cell. Both are stored, runtime-swappable std::functions (the
    // configurable-model-function case, like Lenia1Player — see its header).
    //
    // NOTE: the Java version deduplicated identical kernels within a source
    // column to skip redundant convolutions (FindKernelDuplicates /
    // fftFieldRefs). That optimization is dropped here: it only compared
    // each kernel against its immediate predecessor and, for three or more
    // identical kernels in a column, read a convolution buffer that was
    // never computed that step — a real bug (see docs/JAVA_BUGS_FOUND.md).
    // This port always computes every K_ij*A_j, which is correct and
    // simpler; caching fft(A_j) across targets is a valid future
    // optimization.
    class LeniaNPlayer
    {
    public:
        using KernelFn = std::function<double(int32_t, int32_t, double)>;
        using GrowthFn = std::function<double(int32_t, const std::vector<double> &)>;

        // sideLenExp sets the (square) side length to 2^sideLenExp; nPlayers
        // is the channel count; deltaT is the integration step.
        LeniaNPlayer(int32_t  sideLenExp,
                     int32_t  nPlayers,
                     double   deltaT,
                     KernelFn kernelFn,
                     GrowthFn growthFn);
        ~LeniaNPlayer();

        LeniaNPlayer(const LeniaNPlayer &other) = delete;
        LeniaNPlayer &operator=(const LeniaNPlayer &other) = delete;

        void SetKernelFunction(KernelFn kernelFn);
        void SetGrowthFunction(GrowthFn growthFn);

        // Snapshots the current kernel state (function + every derived
        // per-pair kernel artifact) under label, overwriting any earlier
        // snapshot with the same label; LoadKernel restores it exactly,
        // without recomputing, and throws std::invalid_argument for an
        // unknown label. A snapshot can be loaded any number of times.
        void StoreKernel(const std::string &label);
        void LoadKernel(const std::string &label);

        // Recomputes every (normalized, FFT-transformed) kernel from the
        // current kernel function.
        void RecalcKernels();

        // Advances all channels one step.
        void Update();

        int32_t NPlayers() const;
        int32_t Xdim() const;
        int32_t Ydim() const;
        int32_t Length() const;
        int32_t GetTick() const;
        void ResetTick();

        double Get(int32_t iPlayer,
                   int32_t i) const;
        double Get(int32_t iPlayer,
                   int32_t x,
                   int32_t y) const;
        void Set(int32_t iPlayer,
                 int32_t i,
                 double  val);
        void Set(int32_t iPlayer,
                 int32_t x,
                 int32_t y,
                 double  val);

        const Grid2D<double> &GetField(int32_t iPlayer) const;

        // The convolution K_ij * A_j (target iPlayer, source jPlayer) from
        // the most recent Update.
        double GetConvolvedFieldVal(int32_t iPlayer,
                                    int32_t jPlayer,
                                    int32_t i) const;
        double GetConvolvedFieldVal(int32_t iPlayer,
                                    int32_t jPlayer,
                                    int32_t x,
                                    int32_t y) const;

        // The un-normalized weight of kernel K_ij at a cell, for
        // visualization.
        double GetKernelVal(int32_t iPlayer,
                            int32_t jPlayer,
                            int32_t i) const;
        double GetKernelVal(int32_t iPlayer,
                            int32_t jPlayer,
                            int32_t x,
                            int32_t y) const;

    private:
        int32_t KernelIndex(int32_t iPlayer,
                            int32_t jPlayer) const;
        int32_t ConvolvedIndex(int32_t iPlayer,
                               int32_t jPlayer,
                               int32_t cell) const;

        int32_t m_nPlayers;
        double m_deltaT;
        int32_t m_tick;
        int32_t m_xDim;
        int32_t m_yDim;
        int32_t m_length;
        KernelFn m_kernelFn;
        GrowthFn m_growthFn;
        std::vector<Grid2D<double>> m_fields;
        std::vector<Grid2D<double>> m_kernelFields;
        std::vector<double> m_kSums;
        std::vector<FftGrid> m_fftKernels;
        FftGrid m_fftWork;
        std::vector<double> m_convolved;
        std::vector<double> m_growthScratch;
        std::map<std::string, LeniaKernelSnapshot> m_kernelStorage;
    };
}
