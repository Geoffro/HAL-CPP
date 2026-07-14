#include <doctest/doctest.h>

#include "Hal/Tools/Fft.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Direct O(n^2) DFT with the same unitary (1/sqrt(n)) normalization and
// sign convention as Hal::Fft, used as an independent ground truth (a
// different algorithm entirely, so agreement isn't a shared-bug artifact).
//
// NOTE on the sign: Hal::Fft's "direct" transform applies the twiddle
// factor (cos(arg) - i*sin(arg)) with arg = -2*pi*p/n, which is
// exp(+2*pi*i*p/n) — i.e. its forward direction uses the *positive*
// exponent, the opposite of the textbook default. This is HAL's own
// convention (self-consistent: inverse undoes forward, and the
// convolution theorem holds — see the tests below), not a defect, so the
// reference here matches it: sign = +1 for the direct transform.
static std::vector<std::complex<double>> ReferenceDft(const std::vector<std::complex<double>> &input,
                                                      bool                                     direct)
{
    int32_t n = static_cast<int32_t>(input.size());
    double sign = direct ? 1.0 : -1.0;
    double norm = 1.0 / std::sqrt(static_cast<double>(n));
    std::vector<std::complex<double>> output(static_cast<size_t>(n));
    for (int32_t k = 0; k < n; k++)
    {
        std::complex<double> sum(0.0, 0.0);
        for (int32_t j = 0; j < n; j++)
        {
            double angle = sign * 2.0 * M_PI * k * j / n;
            sum += input[static_cast<size_t>(j)] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        output[static_cast<size_t>(k)] = sum * norm;
    }
    return output;
}

static std::vector<std::complex<double>> Deinterleave(const std::vector<double> &interleaved)
{
    std::vector<std::complex<double>> result(interleaved.size() / 2);
    for (size_t i = 0; i < result.size(); i++)
    {
        result[i] = std::complex<double>(interleaved[i * 2], interleaved[i * 2 + 1]);
    }
    return result;
}

TEST_CASE("Hal::Fft matches a direct DFT for a known signal (forward and inverse)")
{
    std::vector<double> real = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0};
    std::vector<double> imag = {0.5, -1.0, 0.0, 2.0, -0.5, 1.5, 1.0, -2.0};

    std::vector<std::complex<double>> signal(real.size());
    for (size_t i = 0; i < real.size(); i++)
    {
        signal[i] = std::complex<double>(real[i], imag[i]);
    }

    for (bool direct : {true, false})
    {
        std::vector<std::complex<double>> fromFft = Deinterleave(Hal::Fft(real, imag, direct));
        std::vector<std::complex<double>> reference = ReferenceDft(signal, direct);
        REQUIRE(fromFft.size() == reference.size());
        for (size_t i = 0; i < reference.size(); i++)
        {
            CAPTURE(i);
            CAPTURE(direct);
            CHECK(fromFft[i].real() == doctest::Approx(reference[i].real()).epsilon(1e-12));
            CHECK(fromFft[i].imag() == doctest::Approx(reference[i].imag()).epsilon(1e-12));
        }
    }
}

TEST_CASE("Hal::Fft inverse of forward reproduces the original signal (unitary round-trip)")
{
    std::vector<double> interleaved = {
        1.0, 0.0, -2.0, 0.5, 3.0, -1.0, 0.0, 0.0,
        0.25, 4.0, -1.5, -1.5, 2.0, 2.0, -3.0, 1.0};

    std::vector<double> forward = Hal::Fft(interleaved, true);
    std::vector<double> roundTrip = Hal::Fft(forward, false);

    REQUIRE(roundTrip.size() == interleaved.size());
    for (size_t i = 0; i < interleaved.size(); i++)
    {
        CAPTURE(i);
        CHECK(roundTrip[i] == doctest::Approx(interleaved[i]).epsilon(1e-12));
    }
}

TEST_CASE("Hal::Fft of a constant signal concentrates all energy at DC")
{
    const int32_t n = 16;
    std::vector<double> real(n, 1.0);
    std::vector<double> imag(n, 0.0);

    std::vector<std::complex<double>> spectrum = Deinterleave(Hal::Fft(real, imag, true));
    // DC bin == sqrt(n) * mean == sqrt(n); all others zero.
    CHECK(spectrum[0].real() == doctest::Approx(std::sqrt(static_cast<double>(n))).epsilon(1e-12));
    CHECK(spectrum[0].imag() == doctest::Approx(0.0).epsilon(1e-12));
    for (int32_t k = 1; k < n; k++)
    {
        CAPTURE(k);
        CHECK(std::abs(spectrum[static_cast<size_t>(k)]) == doctest::Approx(0.0).epsilon(1e-12));
    }
}

TEST_CASE("Hal::Fft of a unit impulse is flat with magnitude 1/sqrt(n)")
{
    const int32_t n = 8;
    std::vector<double> real(n, 0.0);
    std::vector<double> imag(n, 0.0);
    real[0] = 1.0;

    std::vector<std::complex<double>> spectrum = Deinterleave(Hal::Fft(real, imag, true));
    double expectedMagnitude = 1.0 / std::sqrt(static_cast<double>(n));
    for (int32_t k = 0; k < n; k++)
    {
        CAPTURE(k);
        CHECK(spectrum[static_cast<size_t>(k)].real() == doctest::Approx(expectedMagnitude).epsilon(1e-12));
        CHECK(spectrum[static_cast<size_t>(k)].imag() == doctest::Approx(0.0).epsilon(1e-12));
    }
}

// The actual use case (Lenia kernel convolution): circular convolution via
// the FFT must match a direct circular convolution. With unitary
// normalization, F(a (*) b) = sqrt(n) * F(a) .* F(b), so
// a (*) b == invFft( sqrt(n) * Fft(a) .* Fft(b) ).
TEST_CASE("Hal::Fft implements circular convolution via the convolution theorem")
{
    const int32_t n = 8;
    std::vector<double> a = {1.0, 2.0, 0.0, -1.0, 3.0, 0.5, -2.0, 1.0};
    std::vector<double> b = {0.0, 1.0, 0.5, 0.0, -1.0, 2.0, 1.0, 0.0};

    // Direct circular convolution.
    std::vector<double> directConv(n, 0.0);
    for (int32_t m = 0; m < n; m++)
    {
        double sum = 0.0;
        for (int32_t j = 0; j < n; j++)
        {
            int32_t idx = ((m - j) % n + n) % n;
            sum += a[static_cast<size_t>(j)] * b[static_cast<size_t>(idx)];
        }
        directConv[static_cast<size_t>(m)] = sum;
    }

    std::vector<double> zeros(n, 0.0);
    std::vector<std::complex<double>> fa = Deinterleave(Hal::Fft(a, zeros, true));
    std::vector<std::complex<double>> fb = Deinterleave(Hal::Fft(b, zeros, true));

    double scale = std::sqrt(static_cast<double>(n));
    std::vector<double> productReal(n);
    std::vector<double> productImag(n);
    for (int32_t k = 0; k < n; k++)
    {
        std::complex<double> product = scale * fa[static_cast<size_t>(k)] * fb[static_cast<size_t>(k)];
        productReal[static_cast<size_t>(k)] = product.real();
        productImag[static_cast<size_t>(k)] = product.imag();
    }

    std::vector<std::complex<double>> fftConv = Deinterleave(Hal::Fft(productReal, productImag, false));
    for (int32_t m = 0; m < n; m++)
    {
        CAPTURE(m);
        CHECK(fftConv[static_cast<size_t>(m)].real() == doctest::Approx(directConv[static_cast<size_t>(m)]).epsilon(1e-12));
        CHECK(fftConv[static_cast<size_t>(m)].imag() == doctest::Approx(0.0).epsilon(1e-12));
    }
}

TEST_CASE("Hal::Fft rejects non-power-of-two and mismatched inputs")
{
    std::vector<double> real6(6, 1.0);
    std::vector<double> imag6(6, 0.0);
    CHECK_THROWS_AS(Hal::Fft(real6, imag6, true), std::invalid_argument);

    std::vector<double> real4(4, 1.0);
    std::vector<double> imag3(3, 0.0);
    CHECK_THROWS_AS(Hal::Fft(real4, imag3, true), std::invalid_argument);

    std::vector<double> oddInterleaved(6, 0.0); // 3 complex samples, not a power of two
    CHECK_THROWS_AS(Hal::Fft(oddInterleaved, true), std::invalid_argument);
}
