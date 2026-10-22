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

        // In-place radix-2 FFT on split real/imag arrays, scaling every
        // element by normFactor at the end. Shared by Hal::Fft and
        // Hal::FftGrid (see Fft.h for the sign convention). Throws
        // std::invalid_argument if the length is not a power of two.
        void FftCore(std::vector<double> &xReal,
                     std::vector<double> &xImag,
                     bool                 direct,
                     double               normFactor);
    }
}
