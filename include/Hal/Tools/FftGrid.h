#pragma once

#include "Hal/Grids/Grid2D.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    // 2D complex grid with an in-place FFT, ported from
    // Tools/Lenia/FFTGrid.java. The core use is fast kernel convolution
    // (the mechanism behind Lenia's continuous neighborhood interaction):
    // FFT the state grid and the (same-sized) kernel grid, multiply them
    // elementwise, inverse-FFT, and take the real part.
    //
    // Both dimensions must be powers of two (the constructor throws
    // std::invalid_argument otherwise — the Java original silently assumed
    // it). Complex data is stored as two flat row-major (index =
    // x*yDim + y) arrays, matching Grid2D's own indexing.
    //
    // Normalization is asymmetric, preserving the Java convention exactly:
    // the forward transform (Fft2) is unnormalized and the inverse (Ifft2)
    // divides by the total element count, so Ifft2(Fft2(x)) == x and an
    // FFT-based circular convolution needs no extra scaling — see the tests
    // for both. The 1D transform underneath is the same
    // Hal::Internal::FftCore that Hal::Fft uses (which passes a different,
    // unitary factor); this shares that implementation rather than
    // duplicating it. The (positive-exponent) forward sign convention is
    // likewise shared — see Hal::Fft's header.
    class FftGrid
    {
    public:
        FftGrid(int32_t xDim,
                int32_t yDim);
        explicit FftGrid(const Grid2D<double> &realGrid);
        ~FftGrid();

        FftGrid(const FftGrid &other) = delete;
        FftGrid &operator=(const FftGrid &other) = delete;

        // Movable (so a std::vector<FftGrid> of per-pair kernels can be
        // built, as LeniaNPlayer needs), but not copyable — the complex
        // data arrays are potentially large and copies are rarely wanted.
        // When a copy IS wanted (kernel snapshots — see
        // LeniaNPlayer::StoreKernel), Clone() makes the deep copy
        // explicit at the call site.
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
