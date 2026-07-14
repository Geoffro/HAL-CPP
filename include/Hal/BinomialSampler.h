#pragma once

#include <cstdint>

namespace Hal
{
    class Rand;

    // Binomial-distribution sampler, ported from HAL's
    // Tools/Internal/Binomial.java. Combines three strategies depending on
    // n*p, matching the Java original: direct Bernoulli-trial summation for
    // small n, the "rk_binomial_inversion" algorithm (as used by NumPy) for
    // small n*p, and the BTPE algorithm (Kachitvichyanukul & Schmeiser, via
    // the Colt library) for the general case. BTPE caches intermediate
    // values keyed on the last (n, p) pair, hence the member state.
    //
    // NOTE: the Java source declared `log_p`/`log_q`/`log_n` fields, a
    // `logFactorial`/`logFactorials` helper, `SetNandPInt`/`SetNandPLong`,
    // and no-argument `ColtInt`/`ColtLong` overloads that are never called
    // anywhere in HAL (dead/unreferenced public API) — dropped here rather
    // than ported.
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

        int32_t ColtInt(int32_t  n,
                        double   p,
                        Rand    &rand);

        int64_t ColtLong(int64_t  n,
                         double   p,
                         Rand    &rand);

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
        static double StirlingCorrection(int32_t k);
        static double StirlingCorrection(int64_t k);

        int32_t m_nLast;
        int32_t m_nPrev;
        int64_t m_nLastLong;
        int64_t m_nPrevLong;
        double m_par;
        double m_np;
        double m_p0;
        double m_q;
        double m_pLast;
        double m_pPrev;
        int32_t m_b;
        int32_t m_m;
        int32_t m_nm;
        int64_t m_bLong;
        int64_t m_mLong;
        int64_t m_nmLong;
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
