#include <doctest/doctest.h>

#include "Hal/Grids/Grid2D.h"
#include "Hal/Tools/FftGrid.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

static Hal::Grid2D<double> MakeGrid(int32_t                    xDim,
                                    int32_t                    yDim,
                                    const std::vector<double> &rowMajorValues)
{
    Hal::Grid2D<double> grid(xDim, yDim);
    for (int32_t x = 0; x < xDim; x++)
    {
        for (int32_t y = 0; y < yDim; y++)
        {
            grid.Set(x, y, rowMajorValues[static_cast<size_t>(x * yDim + y)]);
        }
    }
    return grid;
}

TEST_CASE("Hal::FftGrid: Ifft2 of Fft2 reproduces the original grid")
{
    const int32_t xDim = 8;
    const int32_t yDim = 4;
    std::vector<double> values(static_cast<size_t>(xDim * yDim));
    for (size_t i = 0; i < values.size(); i++)
    {
        values[i] = static_cast<double>((i * 7) % 13) - 4.0;
    }
    Hal::Grid2D<double> original = MakeGrid(xDim, yDim, values);

    Hal::FftGrid grid(original);
    grid.Fft2();
    grid.Ifft2();

    for (int32_t x = 0; x < xDim; x++)
    {
        for (int32_t y = 0; y < yDim; y++)
        {
            CAPTURE(x);
            CAPTURE(y);
            CHECK(grid.GetReal(x, y) == doctest::Approx(original.Get(x, y)).epsilon(1e-12));
            CHECK(grid.GetImag(x, y) == doctest::Approx(0.0).epsilon(1e-12));
        }
    }
}

TEST_CASE("Hal::FftGrid: Fft2 of a constant grid puts the total sum in the DC bin")
{
    const int32_t xDim = 4;
    const int32_t yDim = 8;
    std::vector<double> values(static_cast<size_t>(xDim * yDim), 2.5);
    Hal::FftGrid grid(MakeGrid(xDim, yDim, values));
    grid.Fft2();

    // Forward is unnormalized, so DC = sum of all elements.
    double expectedSum = 2.5 * xDim * yDim;
    CHECK(grid.GetReal(0, 0) == doctest::Approx(expectedSum).epsilon(1e-9));
    CHECK(grid.GetImag(0, 0) == doctest::Approx(0.0).epsilon(1e-9));
    for (int32_t x = 0; x < xDim; x++)
    {
        for (int32_t y = 0; y < yDim; y++)
        {
            if ((x == 0) && (y == 0))
            {
                continue;
            }
            CAPTURE(x);
            CAPTURE(y);
            CHECK(grid.GetReal(x, y) == doctest::Approx(0.0).epsilon(1e-9));
            CHECK(grid.GetImag(x, y) == doctest::Approx(0.0).epsilon(1e-9));
        }
    }
}

// The actual Lenia use case: 2D circular convolution of a state grid with
// a kernel, done in the frequency domain, must match a direct 2D circular
// convolution. The asymmetric normalization (raw forward, 1/N inverse)
// means no extra scaling is needed.
TEST_CASE("Hal::FftGrid: FFT-based 2D convolution matches a direct circular convolution")
{
    const int32_t xDim = 8;
    const int32_t yDim = 8;

    std::vector<double> stateValues(static_cast<size_t>(xDim * yDim), 0.0);
    std::vector<double> kernelValues(static_cast<size_t>(xDim * yDim), 0.0);
    for (int32_t x = 0; x < xDim; x++)
    {
        for (int32_t y = 0; y < yDim; y++)
        {
            size_t idx = static_cast<size_t>(x * yDim + y);
            stateValues[idx] = static_cast<double>((x * 3 + y * 5) % 7);
        }
    }
    // A small 3x3-ish kernel placed at the origin (wraps around).
    kernelValues[static_cast<size_t>(0 * yDim + 0)] = 0.5;
    kernelValues[static_cast<size_t>(0 * yDim + 1)] = 0.25;
    kernelValues[static_cast<size_t>(1 * yDim + 0)] = 0.25;
    kernelValues[static_cast<size_t>((xDim - 1) * yDim + 0)] = 0.125;
    kernelValues[static_cast<size_t>(0 * yDim + (yDim - 1))] = 0.125;

    Hal::Grid2D<double> stateGrid = MakeGrid(xDim, yDim, stateValues);
    Hal::Grid2D<double> kernelGrid = MakeGrid(xDim, yDim, kernelValues);

    // Direct 2D circular convolution.
    std::vector<double> directConv(static_cast<size_t>(xDim * yDim), 0.0);
    for (int32_t x = 0; x < xDim; x++)
    {
        for (int32_t y = 0; y < yDim; y++)
        {
            double sum = 0.0;
            for (int32_t dx = 0; dx < xDim; dx++)
            {
                for (int32_t dy = 0; dy < yDim; dy++)
                {
                    int32_t sx = ((x - dx) % xDim + xDim) % xDim;
                    int32_t sy = ((y - dy) % yDim + yDim) % yDim;
                    sum += stateGrid.Get(sx, sy) * kernelGrid.Get(dx, dy);
                }
            }
            directConv[static_cast<size_t>(x * yDim + y)] = sum;
        }
    }

    // Frequency-domain convolution.
    Hal::FftGrid stateFft(stateGrid);
    Hal::FftGrid kernelFft(kernelGrid);
    stateFft.Fft2();
    kernelFft.Fft2();
    stateFft.ComplexMultiply(kernelFft);
    stateFft.Ifft2();

    for (int32_t x = 0; x < xDim; x++)
    {
        for (int32_t y = 0; y < yDim; y++)
        {
            CAPTURE(x);
            CAPTURE(y);
            CHECK(stateFft.GetReal(x, y) == doctest::Approx(directConv[static_cast<size_t>(x * yDim + y)]).epsilon(1e-9));
            CHECK(stateFft.GetImag(x, y) == doctest::Approx(0.0).epsilon(1e-9));
        }
    }
}

TEST_CASE("Hal::FftGrid: FftShift applied twice is the identity, and moves DC to the center")
{
    const int32_t xDim = 4;
    const int32_t yDim = 4;
    std::vector<double> values(static_cast<size_t>(xDim * yDim));
    for (size_t i = 0; i < values.size(); i++)
    {
        values[i] = static_cast<double>(i);
    }
    Hal::Grid2D<double> original = MakeGrid(xDim, yDim, values);

    Hal::FftGrid grid(original);
    grid.FftShift();
    // Corner (0,0) moves to the center (xDim/2, yDim/2).
    CHECK(grid.GetReal(xDim / 2, yDim / 2) == doctest::Approx(original.Get(0, 0)));

    grid.FftShift();
    for (int32_t x = 0; x < xDim; x++)
    {
        for (int32_t y = 0; y < yDim; y++)
        {
            CAPTURE(x);
            CAPTURE(y);
            CHECK(grid.GetReal(x, y) == doctest::Approx(original.Get(x, y)));
        }
    }
}

TEST_CASE("Hal::FftGrid: non-power-of-two dimensions and mismatched operations throw")
{
    CHECK_THROWS_AS(Hal::FftGrid(6, 4), std::invalid_argument);
    CHECK_THROWS_AS(Hal::FftGrid(8, 5), std::invalid_argument);

    Hal::FftGrid a(4, 4);
    Hal::FftGrid b(8, 8);
    CHECK_THROWS_AS(a.ComplexMultiply(b), std::invalid_argument);

    Hal::Grid2D<double> wrongSize(2, 2);
    CHECK_THROWS_AS(a.SetGrid(wrongSize), std::invalid_argument);
}
