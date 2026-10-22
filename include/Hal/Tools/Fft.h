#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Radix-2 complex FFT (Brigham's algorithm). direct selects the forward
    // transform, which uses the positive exponent exp(+2*pi*i*k*n/N) —
    // opposite the textbook default, which matters only when comparing bins
    // with another library. Both directions scale by 1/sqrt(n), so
    // Fft(Fft(x, true), false) == x. n (the number of complex samples) must
    // be a power of two; Fft throws std::invalid_argument otherwise.

    // Interleaved form: `interleaved` is [re0, im0, re1, im1, ...] of length
    // 2*n. Returns the transformed spectrum in the same interleaved layout.
    std::vector<double> Fft(const std::vector<double> &interleaved,
                            bool                       direct);

    // Split form: `real` and `imag` are each length n. Returns the
    // transformed spectrum interleaved as [re0, im0, re1, im1, ...] of
    // length 2*n.
    std::vector<double> Fft(const std::vector<double> &real,
                            const std::vector<double> &imag,
                            bool                       direct);
}
