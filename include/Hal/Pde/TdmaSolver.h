#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Tridiagonal matrix solver (Thomas algorithm), used by ADI diffusion
    // one line at a time. getB/getC/getA give the diagonal, super-diagonal,
    // and sub-diagonal; getIn the right-hand side; setOut receives the
    // solution. TDMAperiodic handles the cyclic system (corner entries) via
    // a Sherman-Morrison rank-1 update solved with two ordinary TDMA calls.
    class TdmaSolver
    {
    public:
        explicit TdmaSolver(int32_t scratchLen)
            : m_scratch(static_cast<size_t>(scratchLen) * kNumScratches, 0.0),
              m_scratchLen(scratchLen)
        {
        }

        ~TdmaSolver()
        {
        }

        template <typename GetInFn, typename SetOutFn, typename GetBFn, typename GetCFn, typename GetAFn>
        void TDMA(int32_t  lenToSolve,
                  GetInFn  getIn,
                  SetOutFn setOut,
                  GetBFn   getB,
                  GetCFn   getC,
                  GetAFn   getA)
        {
            EnsureScratch(lenToSolve);
            int32_t as = 0;
            int32_t bs = m_scratchLen;
            int32_t cs = m_scratchLen * 2;
            int32_t ws = m_scratchLen * 3;
            int32_t ds = m_scratchLen * 4;

            m_scratch[bs] = getB(0);
            m_scratch[cs] = getC(0);
            m_scratch[ds] = getIn(0);
            for (int32_t i = 1; i < lenToSolve; i++)
            {
                if (i < (lenToSolve - 1))
                {
                    m_scratch[cs + i] = getC(i);
                }
                m_scratch[bs + i] = getB(i);
                m_scratch[as + i] = getA(i);
                m_scratch[ds + i] = getIn(i);
                m_scratch[ws + i] = m_scratch[as + i] / m_scratch[bs + i - 1];
                m_scratch[bs + i] = m_scratch[bs + i] - m_scratch[ws + i] * m_scratch[cs + i - 1];
                m_scratch[ds + i] = m_scratch[ds + i] - m_scratch[ws + i] * m_scratch[ds + i - 1];
            }

            int32_t i = lenToSolve - 1;
            double x = m_scratch[ds + i] / m_scratch[bs + i];
            setOut(i, x);
            for (i -= 1; i >= 0; i--)
            {
                x = (m_scratch[ds + i] - m_scratch[cs + i] * x) / m_scratch[bs + i];
                setOut(i, x);
            }
        }

        template <typename GetInFn, typename SetOutFn, typename GetBFn, typename GetCFn, typename GetAFn>
        void TDMAperiodic(int32_t  lenToSolve,
                          GetInFn  getIn,
                          SetOutFn setOut,
                          GetBFn   getB,
                          GetCFn   getC,
                          GetAFn   getA)
        {
            EnsureScratch(lenToSolve);
            int32_t ys = m_scratchLen * 5;
            int32_t zs = m_scratchLen * 6;

            const double gamma = -getB(0);
            const double alpha = getA(0);
            const double beta  = getC(lenToSolve - 1);

            auto adjustedGetB = [&](int32_t i)
            {
                if (i == 0)
                {
                    return getB(i) - gamma;
                }
                if (i == (lenToSolve - 1))
                {
                    return getB(i) - (alpha * beta) / gamma;
                }
                return getB(i);
            };

            TDMA(
                lenToSolve, getIn, [&](int32_t i, double d)
                { m_scratch[ys + i] = d; },
                adjustedGetB, getC, getA);

            TDMA(
                lenToSolve, [&](int32_t i)
                {
                if (i == 0) { return gamma; }
                if (i == (lenToSolve - 1)) { return alpha; }
                return 0.0; },
                [&](int32_t i, double d)
                { m_scratch[zs + i] = d; },
                adjustedGetB, getC, getA);

            double vEnd = beta / gamma;
            double vy = m_scratch[ys] + m_scratch[ys + lenToSolve - 1] * vEnd;
            double vz = m_scratch[zs] + m_scratch[zs + lenToSolve - 1] * vEnd;
            double factor = vy / (1 + vz);
            for (int32_t i = 0; i < lenToSolve; i++)
            {
                setOut(i, m_scratch[ys + i] - factor * m_scratch[zs + i]);
            }
        }

    private:
        void EnsureScratch(int32_t lenToSolve)
        {
            if (m_scratchLen < lenToSolve)
            {
                m_scratchLen = lenToSolve;
                m_scratch.assign(static_cast<size_t>(m_scratchLen) * kNumScratches, 0.0);
            }
        }

        static constexpr int32_t kNumScratches = 7;

        std::vector<double> m_scratch;
        int32_t m_scratchLen;
    };
}
