#include "Hal/BinomialSampler.h"

#include "Hal/Rand.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Hal
{
    namespace Internal
    {
        const double kStirlingCorrectionTable[] =
            {
                0.0,
                8.106146679532726e-02, 4.134069595540929e-02,
                2.767792568499834e-02, 2.079067210376509e-02,
                1.664469118982119e-02, 1.387612882307075e-02,
                1.189670994589177e-02, 1.041126526197209e-02,
                9.255462182712733e-03, 8.330563433362871e-03,
                7.573675487951841e-03, 6.942840107209530e-03,
                6.408994188004207e-03, 5.951370112758848e-03,
                5.554733551962801e-03, 5.207655919609640e-03,
                4.901395948434738e-03, 4.629153749334029e-03,
                4.385560249232324e-03, 4.166319691996922e-03,
                3.967954218640860e-03, 3.787618068444430e-03,
                3.622960224683090e-03, 3.472021382978770e-03,
                3.333155636728090e-03, 3.204970228055040e-03,
                3.086278682608780e-03, 2.976063983550410e-03,
                2.873449362352470e-03, 2.777674929752690e-03};
    }

    double BinomialSampler::StirlingCorrection(int64_t k)
    {
        const double c1 = 8.33333333333333333e-02;
        const double c3 = -2.77777777777777778e-03;
        const double c5 = 7.93650793650793651e-04;
        const double c7 = -5.95238095238095238e-04;

        if (k > 30)
        {
            double r = 1.0 / static_cast<double>(k);
            double rr = r * r;
            return r * (c1 + rr * (c3 + rr * (c5 + rr * c7)));
        }
        return Internal::kStirlingCorrectionTable[k];
    }

    void BinomialSampler::CheckProbability(double p)
    {
        if ((p > 1) || (p < 0) || std::isnan(p))
        {
            throw std::invalid_argument("Probability argument cannot be > 1 or < 0");
        }
    }

    BinomialSampler::BinomialSampler()
        : m_nLast(-1),
          m_nPrev(-1),
          m_par(0),
          m_np(0),
          m_p0(0),
          m_q(0),
          m_pLast(-1.0),
          m_pPrev(-1.0),
          m_b(0),
          m_m(0),
          m_nm(0),
          m_pq(0),
          m_rc(0),
          m_ss(0),
          m_xm(0),
          m_xl(0),
          m_xr(0),
          m_ll(0),
          m_lr(0),
          m_c(0),
          m_p1(0),
          m_p2(0),
          m_p3(0),
          m_p4(0),
          m_ch(0)
    {
    }

    BinomialSampler::~BinomialSampler()
    {
    }

    int32_t BinomialSampler::BinomialProbSample(int32_t  n,
                                                double   p,
                                                Rand    &rand)
    {
        int32_t total = 0;
        for (int32_t i = 0; i < n; i++)
        {
            if (rand.Double() < p)
            {
                total++;
            }
        }
        return total;
    }

    int64_t BinomialSampler::BinomialProbSample(int64_t  n,
                                                double   p,
                                                Rand    &rand)
    {
        int64_t total = 0;
        for (int64_t i = 0; i < n; i++)
        {
            if (rand.Double() < p)
            {
                total++;
            }
        }
        return total;
    }

    int32_t BinomialSampler::RkBinomialInversion(int32_t  n,
                                                 double   p,
                                                 Rand    &rand)
    {
        // see rk_binomial_inversion at
        // https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/distributions.c
        // for use when p*n < 30
        double q = 1.0 - p;
        double qn = std::exp(n * std::log(q));
        double np = n * p;
        int32_t bound = static_cast<int32_t>(std::min(static_cast<double>(n), np + 10.0 * std::sqrt(np * q + 1)));

        int32_t x = 0;
        double px = qn;

        double u = rand.Double();
        while (u > px)
        {
            x++;
            if (x > bound)
            {
                x = 0;
                px = qn;
                u = rand.Double();
            }
            else
            {
                u -= px;
                px = ((n - x + 1) * p * px) / (x * q);
            }
        }
        return x;
    }

    int64_t BinomialSampler::RkBinomialInversion(int64_t  n,
                                                 double   p,
                                                 Rand    &rand)
    {
        double q = 1.0 - p;
        double qn = std::exp(static_cast<double>(n) * std::log(q));
        double np = static_cast<double>(n) * p;
        int64_t bound = static_cast<int64_t>(std::min(static_cast<double>(n), np + 10.0 * std::sqrt(np * q + 1)));

        int64_t x = 0;
        double px = qn;

        double u = rand.Double();
        while (u > px)
        {
            x++;
            if (x > bound)
            {
                x = 0;
                px = qn;
                u = rand.Double();
            }
            else
            {
                u -= px;
                px = ((n - x + 1) * p * px) / (x * q);
            }
        }
        return x;
    }

    int32_t BinomialSampler::ColtInt(int32_t  n,
                                     double   p,
                                     Rand    &rand)
    {
        return static_cast<int32_t>(ColtLong(n, p, rand));
    }

    int64_t BinomialSampler::ColtLong(int64_t  n,
                                      double   p,
                                      Rand    &rand)
    {
        if ((n != m_nLast) || (p != m_pLast))
        {
            m_nLast = n;
            m_pLast = p;
            m_par = std::min(p, 1.0 - p);
            m_q = 1.0 - m_par;
            m_np = static_cast<double>(n) * m_par;
            if (m_np <= 0.0)
            {
                return -1;
            }

            double rm = m_np + m_par;
            m_m = static_cast<int64_t>(rm);
            if (m_np < 10.0)
            {
                m_p0 = std::exp(static_cast<double>(n) * std::log(m_q));
                int64_t bh = static_cast<int64_t>(m_np + 10.0 * std::sqrt(m_np * m_q));
                m_b = std::min(n, bh);
            }
            else
            {
                m_pq = m_par / m_q;
                m_rc = (static_cast<double>(n) + 1.0) * m_pq;
                m_ss = m_np * m_q;
                int64_t i = static_cast<int64_t>(2.195 * std::sqrt(m_ss) - 4.6 * m_q);
                m_xm = static_cast<double>(m_m) + 0.5;
                m_xl = static_cast<double>(m_m - i);
                m_xr = static_cast<double>((m_m + i) + 1);
                double f = (rm - m_xl) / (rm - m_xl * m_par);
                m_ll = f * (1.0 + 0.5 * f);
                f = (m_xr - rm) / (m_xr * m_q);
                m_lr = f * (1.0 + 0.5 * f);
                m_c = 0.134 + 20.5 / (15.3 + static_cast<double>(m_m));
                m_p1 = static_cast<double>(i) + 0.5;
                m_p2 = m_p1 * (1.0 + m_c + m_c);
                m_p3 = m_p2 + m_c / m_ll;
                m_p4 = m_p3 + m_c / m_lr;
            }
        }

        int64_t k;
        double u;
        if (m_np < 10.0)
        {
            k = 0;
            double pk = m_p0;
            u = rand.Double();

            while (u > pk)
            {
                k++;
                if (k > m_b)
                {
                    u = rand.Double();
                    k = 0;
                    pk = m_p0;
                }
                else
                {
                    u -= pk;
                    pk = static_cast<double>(n - k + 1) * m_par * pk / (static_cast<double>(k) * m_q);
                }
            }

            return (p > 0.5) ? (n - k) : k;
        }

        while (true)
        {
            double v;
            while (true)
            {
                v = rand.Double();
                u = rand.Double() * m_p4;
                if (u <= m_p1)
                {
                    k = static_cast<int64_t>(m_xm - u + m_p1 * v);
                    return (p > 0.5) ? (n - k) : k;
                }

                double x;
                if (u <= m_p2)
                {
                    x = m_xl + (u - m_p1) / m_c;
                    v = v * m_c + 1.0 - std::abs(m_xm - x) / m_p1;
                    if (v < 1.0)
                    {
                        k = static_cast<int64_t>(x);
                        break;
                    }
                }
                else if (u <= m_p3)
                {
                    x = m_xl + std::log(v) / m_ll;
                    if (x >= 0.0)
                    {
                        k = static_cast<int64_t>(x);
                        v *= (u - m_p2) * m_ll;
                        break;
                    }
                }
                else
                {
                    k = static_cast<int64_t>(m_xr - std::log(v) / m_lr);
                    if (k <= n)
                    {
                        v *= (u - m_p3) * m_lr;
                        break;
                    }
                }
            }

            int64_t km = std::abs(k - m_m);
            if ((km > 20) && (static_cast<double>((km + km) + 2) < m_ss))
            {
                v = std::log(v);
                double t = static_cast<double>(-km * km) / (m_ss + m_ss);
                double e = static_cast<double>(km) / m_ss * ((static_cast<double>(km) * (static_cast<double>(km) * 0.3333333333333333 + 0.625) + 0.16666666666666666) / m_ss + 0.5);
                if (v <= t - e)
                {
                    break;
                }
                if (v <= t + e)
                {
                    if ((n != m_nPrev) || (m_par != m_pPrev))
                    {
                        m_nPrev = n;
                        m_pPrev = m_par;
                        m_nm = n - m_m + 1;
                        m_ch = m_xm * std::log((static_cast<double>(m_m) + 1.0) / (m_pq * static_cast<double>(m_nm))) + StirlingCorrection(m_m + 1) + StirlingCorrection(m_nm);
                    }

                    int64_t nk = n - k + 1;
                    if (v <= m_ch + (static_cast<double>(n) + 1.0) * std::log(static_cast<double>(m_nm) / static_cast<double>(nk)) + (static_cast<double>(k) + 0.5) * std::log(static_cast<double>(nk) * m_pq / (static_cast<double>(k) + 1.0)) - StirlingCorrection(k + 1) - StirlingCorrection(nk))
                    {
                        break;
                    }
                }
            }
            else
            {
                double f = 1.0;
                if (m_m < k)
                {
                    int64_t i = m_m;
                    while (i < k)
                    {
                        i++;
                        f *= m_rc / static_cast<double>(i) - m_pq;
                        if (f < v)
                        {
                            break;
                        }
                    }
                }
                else
                {
                    int64_t i = k;
                    while (i < m_m)
                    {
                        i++;
                        v *= m_rc / static_cast<double>(i) - m_pq;
                        if (v > f)
                        {
                            break;
                        }
                    }
                }

                if (v <= f)
                {
                    break;
                }
            }
        }

        return (p > 0.5) ? (n - k) : k;
    }

    int32_t BinomialSampler::SampleIntFast(int32_t  n,
                                           double   p,
                                           Rand    &rand)
    {
        CheckProbability(p);
        if (p == 1)
        {
            return n;
        }
        if ((p == 0) || (n == 0))
        {
            return 0;
        }
        if (n < 10)
        {
            return BinomialProbSample(n, p, rand);
        }
        if ((p < 0.5) && (n * p < 30))
        {
            return RkBinomialInversion(n, p, rand);
        }
        if ((p >= 0.5) && (n * (1 - p) < 30))
        {
            return n - RkBinomialInversion(n, (1 - p), rand);
        }
        return ColtInt(n, p, rand);
    }

    int32_t BinomialSampler::SampleInt(int32_t  n,
                                       double   p,
                                       Rand    &rand)
    {
        CheckProbability(p);
        if (p == 1)
        {
            return n;
        }
        if ((p == 0) || (n == 0))
        {
            return 0;
        }
        if ((p < 0.5) && (n * p < 30))
        {
            return RkBinomialInversion(n, p, rand);
        }
        if ((p >= 0.5) && (n * (1 - p) < 30))
        {
            return n - RkBinomialInversion(n, (1 - p), rand);
        }
        return ColtInt(n, p, rand);
    }

    int64_t BinomialSampler::SampleLongFast(int64_t  n,
                                            double   p,
                                            Rand    &rand)
    {
        CheckProbability(p);
        if (p == 1)
        {
            return n;
        }
        if ((p == 0) || (n == 0))
        {
            return 0;
        }
        if (n < 10)
        {
            return BinomialProbSample(n, p, rand);
        }
        if ((p < 0.5) && (static_cast<double>(n) * p < 30))
        {
            return RkBinomialInversion(n, p, rand);
        }
        if ((p >= 0.5) && (static_cast<double>(n) * (1 - p) < 30))
        {
            return n - RkBinomialInversion(n, (1 - p), rand);
        }
        return ColtLong(n, p, rand);
    }

    int64_t BinomialSampler::SampleLong(int64_t  n,
                                        double   p,
                                        Rand    &rand)
    {
        CheckProbability(p);
        if (p == 1)
        {
            return n;
        }
        if ((p == 0) || (n == 0))
        {
            return 0;
        }
        if ((p < 0.5) && (static_cast<double>(n) * p < 30))
        {
            return RkBinomialInversion(n, p, rand);
        }
        if ((p >= 0.5) && (static_cast<double>(n) * (1 - p) < 30))
        {
            return n - RkBinomialInversion(n, (1 - p), rand);
        }
        return ColtLong(n, p, rand);
    }
}
