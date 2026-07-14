#include "Hal/Tools/OdeSolver.h"

#include <cstring>

namespace Hal
{
    void OdeSolver::SetupScratch(int32_t length,
                                 int32_t num)
    {
        if (static_cast<int32_t>(m_scratch.size()) < num)
        {
            m_scratch.resize(static_cast<size_t>(num));
        }
        for (int32_t i = 0; i < num; i++)
        {
            if (static_cast<int32_t>(m_scratch[i].size()) != length)
            {
                m_scratch[i].assign(static_cast<size_t>(length), 0.0);
            }
        }
    }

    void OdeSolver::EulerAdd(const double *pState,
                             const double *pDy,
                             double        dt,
                             double       *pOut,
                             int32_t       length)
    {
        for (int32_t i = 0; i < length; i++)
        {
            pOut[i] = pState[i] + pDy[i] * dt;
        }
    }

    void OdeSolver::TempAdd(const double *pA,
                            double        scale,
                            int32_t       length)
    {
        double *pS0 = m_scratch[0].data();
        for (int32_t i = 0; i < length; i++)
        {
            pS0[i] += pA[i] * scale;
        }
    }

    void OdeSolver::CopyState(const double *pState,
                              double       *pOut,
                              int32_t       length)
    {
        std::memcpy(pOut, pState, static_cast<size_t>(length) * sizeof(double));
    }
}
