#pragma once

#include "Hal/Grids/GridGeometry2D.h"
#include "Hal/MathUtil.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

namespace Hal
{
    // 2D value grid of T (Grid2D<double>, Grid2D<int32_t>, ...). Values are
    // read and written directly; there is no delta buffer (see PDEGrid2D).
    template <typename T>
    class Grid2D : public GridGeometry2D
    {
    public:
        Grid2D(int32_t xDim,
               int32_t yDim)
            : Grid2D(xDim, yDim, false, false)
        {
        }

        Grid2D(int32_t xDim,
               int32_t yDim,
               bool    wrapX,
               bool    wrapY)
            : GridGeometry2D(xDim, yDim, wrapX, wrapY),
              m_field(static_cast<size_t>(xDim) * static_cast<size_t>(yDim), T{})
        {
        }

        ~Grid2D()
        {
        }

        T Get(int32_t i) const
        {
            return m_field[i];
        }

        T Get(int32_t x,
              int32_t y) const
        {
            return m_field[I(x, y)];
        }

        const std::vector<T> &GetField() const
        {
            return m_field;
        }

        std::vector<T> &GetField()
        {
            return m_field;
        }

        void Set(int32_t i,
                 T       val)
        {
            m_field[i] = val;
        }

        void Set(int32_t x,
                 int32_t y,
                 T       val)
        {
            m_field[I(x, y)] = val;
        }

        void Add(int32_t i,
                 T       val)
        {
            m_field[i] += val;
        }

        void Add(int32_t x,
                 int32_t y,
                 T       val)
        {
            m_field[I(x, y)] += val;
        }

        // The factor is always a double; integer grids truncate the result.
        void Scale(int32_t i,
                   double  val)
        {
            m_field[i] = static_cast<T>(m_field[i] * val);
        }

        void Scale(int32_t x,
                   int32_t y,
                   double  val)
        {
            Scale(I(x, y), val);
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

        // Integer grids return the truncated average.
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
