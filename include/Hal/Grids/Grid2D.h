#pragma once

#include "Hal/Grids/GridGeometry2D.h"
#include "Hal/MathUtil.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

namespace Hal
{
    // Templated 2D value grid, replacing HAL's per-type Grid2Ddouble /
    // Grid2Dint / Grid2Dlong classes (a duplication Java needed because it
    // has no usable generic primitive arrays) with a single template — see
    // docs/DESIGN.md "Improvements over the Java version". Instantiate as
    // Grid2D<double>, Grid2D<int32_t>, Grid2D<int64_t>, etc.
    //
    // NOTE: Gradient methods (GradientX/GradientY) are deferred to Phase 3,
    // since in Java they depend on PDEequations, which doesn't exist yet.
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

        // Matches HAL's Java Scale, which always takes a double factor
        // (even on the int/long grid types) since scaling by a fractional
        // amount is the common case.
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

        // Integer-valued grids truncate like HAL's Java Grid2Dint.GetAvg;
        // floating-point grids return the true average.
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
