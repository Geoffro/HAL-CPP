#pragma once

#include <cstdint>

namespace Hal
{
    class Rand;

    // Binomial sampler combining three strategies by n and n*p: summed
    // Bernoulli trials for small n (the *Fast variants only), inversion
    // (NumPy's rk_binomial_inversion) when n*p or n*(1-p) is below 30, and
    // BTPE (Kachitvichyanukul & Schmeiser) otherwise. BTPE caches its setup
    // constants for the most recent (n, p).
    class BinomialSampler
    {
    public:
        BinomialSampler();
        ~BinomialSampler();

        int32_t BinomialProbSample(int32_t  n,
                                   double   p,
                                   Rand    &rand);

        int64_t BinomialProbSample(int64_t  n,
                                   double   p,
                                   Rand    &rand);

        int32_t RkBinomialInversion(int32_t  n,
                                    double   p,
                                    Rand    &rand);

        int64_t RkBinomialInversion(int64_t  n,
                                    double   p,
                                    Rand    &rand);

        // BTPE. Returns -1 if n*min(p, 1-p) is not positive.
        int32_t ColtInt(int32_t  n,
                        double   p,
                        Rand    &rand);

        int64_t ColtLong(int64_t  n,
                         double   p,
                         Rand    &rand);

        // The Sample* methods throw std::invalid_argument if p is outside
        // [0, 1].
        int32_t SampleIntFast(int32_t  n,
                              double   p,
                              Rand    &rand);

        int32_t SampleInt(int32_t  n,
                          double   p,
                          Rand    &rand);

        int64_t SampleLongFast(int64_t  n,
                               double   p,
                               Rand    &rand);

        int64_t SampleLong(int64_t  n,
                           double   p,
                           Rand    &rand);

    private:
        static double StirlingCorrection(int64_t k);
        static void CheckProbability(double p);

        int64_t m_nLast;
        int64_t m_nPrev;
        double m_par;
        double m_np;
        double m_p0;
        double m_q;
        double m_pLast;
        double m_pPrev;
        int64_t m_b;
        int64_t m_m;
        int64_t m_nm;
        double m_pq;
        double m_rc;
        double m_ss;
        double m_xm;
        double m_xl;
        double m_xr;
        double m_ll;
        double m_lr;
        double m_c;
        double m_p1;
        double m_p2;
        double m_p3;
        double m_p4;
        double m_ch;
    };
}
