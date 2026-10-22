#pragma once

#include "Hal/Grids/Grid2D.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    // 2D complex grid with an in-place FFT, used for fast circular
    // convolution: FFT the state and a same-sized kernel, multiply
    // elementwise, inverse-FFT, and take the real part. Both dimensions
    // must be powers of two (the constructor throws std::invalid_argument
    // otherwise). Data is stored as real and imaginary arrays indexed
    // x*yDim + y, like Grid2D. Fft2 is unnormalized and Ifft2 divides by
    // the element count, so convolution needs no extra scaling. Sign
    // convention as in Fft.h.
    class FftGrid
    {
    public:
        FftGrid(int32_t xDim,
                int32_t yDim);
        explicit FftGrid(const Grid2D<double> &realGrid);
        ~FftGrid();

        FftGrid(const FftGrid &other) = delete;
        FftGrid &operator=(const FftGrid &other) = delete;

        // Movable but not copyable; Clone() makes an explicit deep copy.
        FftGrid(FftGrid &&other) noexcept;
        FftGrid &operator=(FftGrid &&other) noexcept;

        FftGrid Clone() const;

        int32_t Xdim() const;
        int32_t Ydim() const;

        double GetReal(int32_t x,
                       int32_t y) const;
        double GetImag(int32_t x,
                       int32_t y) const;

        const std::vector<double> &Real() const;
        const std::vector<double> &Imag() const;

        // Overwrites the real part with realGrid and zeroes the imaginary
        // part. realGrid must match this grid's dimensions.
        void SetGrid(const Grid2D<double> &realGrid);

        // Forward 2D FFT (unnormalized), in place.
        void Fft2();

        // Inverse 2D FFT (divided by the element count), in place.
        void Ifft2();

        // Swaps diagonally opposite quadrants — moves the zero-frequency
        // component to the grid center (and back, since applying it twice
        // is the identity for even dimensions). Used to center a kernel's
        // spectrum before convolution.
        void FftShift();

        // Elementwise complex multiply by other (same dimensions), in
        // place — the frequency-domain step of a convolution.
        void ComplexMultiply(const FftGrid &other);

        // Extracts the current real part as a plain value grid.
        Grid2D<double> ToRealGrid() const;

    private:
        void TransformRows(bool   direct,
                           double normFactor);
        void TransformColumns(bool   direct,
                              double normFactor);

        int32_t Index(int32_t x,
                      int32_t y) const;

        int32_t m_xDim;
        int32_t m_yDim;
        std::vector<double> m_real;
        std::vector<double> m_imag;
    };
}
