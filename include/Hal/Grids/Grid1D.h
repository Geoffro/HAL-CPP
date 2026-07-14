#pragma once

#include "Hal/Grids/GridGeometry1D.h"
#include "Hal/MathUtil.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace Hal
{
    // Templated 1D value grid — see Grid2D.h for the rationale behind
    // collapsing HAL's per-type Grid1Ddouble/Grid1Dint/etc. into one
    // template.
    //
    // NOTE: Gradient methods are deferred to Phase 3 (PDEequations).
    template <typename T>
    class Grid1D : public GridGeometry1D
    {
    public:
        explicit Grid1D(int32_t xDim)
            : Grid1D(xDim, false)
        {
        }

        Grid1D(int32_t xDim,
               bool    wrapX)
            : GridGeometry1D(xDim, wrapX),
              m_field(static_cast<size_t>(xDim), T{})
        {
        }

        ~Grid1D()
        {
        }

        T Get(int32_t x) const
        {
            return m_field[x];
        }

        const std::vector<T> &GetField() const
        {
            return m_field;
        }

        std::vector<T> &GetField()
        {
            return m_field;
        }

        void Set(int32_t x,
                 T       val)
        {
            m_field[x] = val;
        }

        void Add(int32_t x,
                 T       val)
        {
            m_field[x] += val;
        }

        void Scale(int32_t x,
                   double  val)
        {
            m_field[x] = static_cast<T>(m_field[x] * val);
        }

        void BoundAll(T min,
                      T max)
        {
            for (int32_t i = 0; i < Length(); i++)
            {
                m_field[i] = Bound(m_field[i], min, max);
            }
        }

        void SetAll(T val)
        {
            std::fill(m_field.begin(), m_field.end(), val);
        }

        void AddAll(T val)
        {
            for (int32_t i = 0; i < Length(); i++)
            {
                m_field[i] += val;
            }
        }

        void MulAll(double val)
        {
            for (int32_t i = 0; i < Length(); i++)
            {
                Scale(i, val);
            }
        }

        void SetAll(const T *pVals)
        {
            std::copy(pVals, pVals + Length(), m_field.begin());
        }

        T GetSum() const
        {
            T tot = T{};
            for (int32_t i = 0; i < Length(); i++)
            {
                tot += m_field[i];
            }
            return tot;
        }

        T GetAvg() const
        {
            T tot = T{};
            for (int32_t i = 0; i < Length(); i++)
            {
                tot += m_field[i];
            }
            return tot / static_cast<T>(Length());
        }

        T GetMax() const
        {
            T max = std::numeric_limits<T>::lowest();
            for (int32_t i = 0; i < Length(); i++)
            {
                max = std::max(m_field[i], max);
            }
            return max;
        }

        T GetMin() const
        {
            T min = std::numeric_limits<T>::max();
            for (int32_t i = 0; i < Length(); i++)
            {
                min = std::min(m_field[i], min);
            }
            return min;
        }

    private:
        std::vector<T> m_field;
    };
}
