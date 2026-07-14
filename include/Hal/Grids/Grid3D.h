#pragma once

#include "Hal/Grids/GridGeometry3D.h"
#include "Hal/MathUtil.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace Hal
{
    // Templated 3D value grid — see Grid2D.h for the rationale behind
    // collapsing HAL's per-type Grid3Ddouble/Grid3Dint/etc. into one
    // template.
    //
    // NOTE: Gradient methods are deferred to Phase 3 (PDEequations).
    template <typename T>
    class Grid3D : public GridGeometry3D
    {
    public:
        Grid3D(int32_t xDim,
               int32_t yDim,
               int32_t zDim)
            : Grid3D(xDim, yDim, zDim, false, false, false)
        {
        }

        Grid3D(int32_t xDim,
               int32_t yDim,
               int32_t zDim,
               bool    wrapX,
               bool    wrapY,
               bool    wrapZ)
            : GridGeometry3D(xDim, yDim, zDim, wrapX, wrapY, wrapZ),
              m_field(static_cast<size_t>(xDim) * static_cast<size_t>(yDim) * static_cast<size_t>(zDim), T{})
        {
        }

        ~Grid3D()
        {
        }

        T Get(int32_t i) const
        {
            return m_field[i];
        }

        T Get(int32_t x,
              int32_t y,
              int32_t z) const
        {
            return m_field[I(x, y, z)];
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
                 int32_t z,
                 T       val)
        {
            m_field[I(x, y, z)] = val;
        }

        void Add(int32_t i,
                 T       val)
        {
            m_field[i] += val;
        }

        void Add(int32_t x,
                 int32_t y,
                 int32_t z,
                 T       val)
        {
            m_field[I(x, y, z)] += val;
        }

        void Scale(int32_t i,
                   double  val)
        {
            m_field[i] = static_cast<T>(m_field[i] * val);
        }

        void Scale(int32_t x,
                   int32_t y,
                   int32_t z,
                   double  val)
        {
            Scale(I(x, y, z), val);
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
