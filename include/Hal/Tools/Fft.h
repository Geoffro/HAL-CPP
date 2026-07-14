#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Radix-2 complex Fast Fourier Transform, ported from
    // Tools/Lenia/FFTbase.java (Orlando Selenu's implementation of the
    // Brigham algorithm). Underpins the FFT-based kernel convolution the
    // Lenia continuous-cellular-automata support is built on.
    //
    // Unitary normalization: both the forward and inverse transforms scale
    // by 1/sqrt(n), so Fft(Fft(x, true), false) == x. `direct` selects the
    // forward transform (direct == true) or the inverse (direct == false).
    //
    // Convention note: this port preserves the Java original's sign
    // convention exactly, in which the "direct" transform's twiddle works
    // out to exp(+2*pi*i*...) — the positive exponent, opposite the
    // textbook default. This is self-consistent (inverse undoes forward,
    // and the convolution theorem holds), just HAL's choice of which
    // direction is "forward"; it only matters if you compare bin-by-bin
    // against another library's forward transform.
    //
    // n (the number of complex samples) must be a power of two; Fft throws
    // std::invalid_argument otherwise. (The Java original printed a message
    // to stdout and returned null — replaced here with an exception per the
    // error-handling standard.)

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
