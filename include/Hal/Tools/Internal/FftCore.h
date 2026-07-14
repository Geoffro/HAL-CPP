#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    namespace Internal
    {
        // log2(n), or -1 if n is not a positive power of two. Shared by the
        // 1D FFT and the 2D FftGrid for up-front size validation.
        int32_t Log2PowerOfTwo(int32_t n);

        // In-place radix-2 FFT on split real/imag arrays, applying the given
        // per-element normalization factor at the end. The transform core
        // shared by Hal::Fft (which passes 1/sqrt(n) for its unitary
        // convention) and Hal::FftGrid (which passes the asymmetric
        // forward-raw / inverse-1/n factors the Java FFTGrid uses). Throws
        // std::invalid_argument if the array length is not a power of two.
        // direct == true is the forward transform; see Hal::Fft's header for
        // the (positive-exponent) sign convention both share.
        void FftCore(std::vector<double> &xReal,
                     std::vector<double> &xImag,
                     bool                 direct,
                     double               normFactor);
    }
}
