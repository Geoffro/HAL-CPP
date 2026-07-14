#include "Hal/Tools/Fft.h"

#include "Hal/Tools/Internal/FftCore.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace Hal
{
    namespace Internal
    {
        // Reverses the low nu bits of j — the bit-reversal permutation the
        // radix-2 FFT reorders its output by.
        int32_t BitReverse(int32_t j,
                           int32_t nu)
        {
            int32_t j1 = j;
            int32_t k = 0;
            for (int32_t i = 1; i <= nu; i++)
            {
                int32_t j2 = j1 / 2;
                k = 2 * k + j1 - 2 * j2;
                j1 = j2;
            }
            return k;
        }

        int32_t Log2PowerOfTwo(int32_t n)
        {
            if (n < 1)
            {
                return -1;
            }
            int32_t nu = 0;
            int32_t remaining = n;
            while (remaining > 1)
            {
                if ((remaining % 2) != 0)
                {
                    return -1;
                }
                remaining /= 2;
                nu++;
            }
            return nu;
        }

        void FftCore(std::vector<double> &xReal,
                     std::vector<double> &xImag,
                     bool                 direct,
                     double               normFactor)
        {
            int32_t n = static_cast<int32_t>(xReal.size());
            int32_t nu = Log2PowerOfTwo(n);
            if (nu < 0)
            {
                throw std::invalid_argument("Hal::Fft requires the number of complex samples to be a power of two");
            }

            int32_t n2 = n / 2;
            int32_t nu1 = nu - 1;
            double constant = direct ? (-2.0 * M_PI) : (2.0 * M_PI);

            int32_t k = 0;
            for (int32_t l = 1; l <= nu; l++)
            {
                while (k < n)
                {
                    for (int32_t i = 1; i <= n2; i++)
                    {
                        double p = static_cast<double>(BitReverse(k >> nu1, nu));
                        double arg = constant * p / n;
                        double c = std::cos(arg);
                        double s = std::sin(arg);
                        double tReal = xReal[k + n2] * c + xImag[k + n2] * s;
                        double tImag = xImag[k + n2] * c - xReal[k + n2] * s;
                        xReal[k + n2] = xReal[k] - tReal;
                        xImag[k + n2] = xImag[k] - tImag;
                        xReal[k] += tReal;
                        xImag[k] += tImag;
                        k++;
                    }
                    k += n2;
                }
                k = 0;
                nu1--;
                n2 /= 2;
            }

            k = 0;
            while (k < n)
            {
                int32_t r = BitReverse(k, nu);
                if (r > k)
                {
                    std::swap(xReal[k], xReal[r]);
                    std::swap(xImag[k], xImag[r]);
                }
                k++;
            }

            for (int32_t i = 0; i < n; i++)
            {
                xReal[i] *= normFactor;
                xImag[i] *= normFactor;
            }
        }

        std::vector<double> Interleave(const std::vector<double> &real,
                                       const std::vector<double> &imag)
        {
            std::vector<double> result(real.size() * 2);
            for (size_t i = 0; i < real.size(); i++)
            {
                result[i * 2] = real[i];
                result[i * 2 + 1] = imag[i];
            }
            return result;
        }
    }

    std::vector<double> Fft(const std::vector<double> &interleaved,
                            bool                       direct)
    {
        if ((interleaved.size() % 2) != 0)
        {
            throw std::invalid_argument("Hal::Fft interleaved input length must be even (re,im pairs)");
        }
        size_t n = interleaved.size() / 2;
        std::vector<double> real(n);
        std::vector<double> imag(n);
        for (size_t i = 0; i < n; i++)
        {
            real[i] = interleaved[i * 2];
            imag[i] = interleaved[i * 2 + 1];
        }
        double unitaryFactor = (n == 0) ? 1.0 : (1.0 / std::sqrt(static_cast<double>(n)));
        Internal::FftCore(real, imag, direct, unitaryFactor);
        return Internal::Interleave(real, imag);
    }

    std::vector<double> Fft(const std::vector<double> &real,
                            const std::vector<double> &imag,
                            bool                       direct)
    {
        if (real.size() != imag.size())
        {
            throw std::invalid_argument("Hal::Fft real and imag arrays must be the same length");
        }
        std::vector<double> workReal = real;
        std::vector<double> workImag = imag;
        double unitaryFactor = (real.empty()) ? 1.0 : (1.0 / std::sqrt(static_cast<double>(real.size())));
        Internal::FftCore(workReal, workImag, direct, unitaryFactor);
        return Internal::Interleave(workReal, workImag);
    }
}
