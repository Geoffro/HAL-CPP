#include "Hal/Grids/GridGeometry1D.h"

#include <cmath>
#include <stdexcept>

namespace Hal
{
    GridGeometry1D::GridGeometry1D(int32_t xDim,
                                   bool    wrapX)
        : m_xDim(xDim),
          m_length(xDim),
          m_wrapX(wrapX)
    {
    }

    GridGeometry1D::~GridGeometry1D()
    {
    }

    int32_t GridGeometry1D::Xdim() const
    {
        return m_xDim;
    }

    int32_t GridGeometry1D::Length() const
    {
        return m_length;
    }

    bool GridGeometry1D::IsWrapX() const
    {
        return m_wrapX;
    }

    int32_t GridGeometry1D::WrapI(int32_t x) const
    {
        if (In(x))
        {
            return x;
        }
        if (m_wrapX)
        {
            return Wrap(x, m_xDim);
        }
        throw std::invalid_argument("cannot map to index in bounds");
    }

    double GridGeometry1D::WrapX(double x) const
    {
        return Wrap(x, static_cast<double>(m_xDim));
    }

    int32_t GridGeometry1D::WrapX(int32_t x) const
    {
        return Wrap(x, m_xDim);
    }

    bool GridGeometry1D::In(int32_t x) const
    {
        return (x >= 0) && (x < m_xDim);
    }

    bool GridGeometry1D::In(double x) const
    {
        return In(static_cast<int32_t>(std::floor(x)));
    }

    bool GridGeometry1D::InWrap(int32_t x) const
    {
        return m_wrapX || InDim(x, m_xDim);
    }

    bool GridGeometry1D::InWrap(double x) const
    {
        return m_wrapX || InDim(x, static_cast<double>(m_xDim));
    }

    double GridGeometry1D::DispX(double x1,
                                 double x2) const
    {
        if (m_wrapX)
        {
            return DispWrap(x2, x1, m_xDim);
        }
        return x2 - x1;
    }

    double GridGeometry1D::Dist(double x1,
                                double x2) const
    {
        return std::abs(DispX(x1, x2));
    }

    double GridGeometry1D::Dist(int32_t x1,
                                int32_t x2) const
    {
        return std::abs(DispX(x1, x2));
    }

    double GridGeometry1D::DistSquared(double x1,
                                       double x2) const
    {
        double dist = DispX(x1, x2);
        return dist * dist;
    }

    int32_t GridGeometry1D::MapXsq(int32_t               x,
                                   const GridGeometry1D &other) const
    {
        return static_cast<int32_t>(((x + 0.5) * other.Xdim()) / m_xDim);
    }

    double GridGeometry1D::MapXpt(double                x,
                                  const GridGeometry1D &other) const
    {
        return x * other.Xdim() / m_xDim;
    }

    int32_t GridGeometry1D::MapHood(int32_t *pHood,
                                    int32_t  hoodLength,
                                    int32_t  centerX) const
    {
        int32_t ptCt = 0;
        int32_t iStart = hoodLength / 2;
        for (int32_t i = iStart; i < hoodLength; i++)
        {
            int32_t x = pHood[i] + centerX;
            if (!InDim(x, m_xDim))
            {
                if (m_wrapX)
                {
                    x = Wrap(x, m_xDim);
                }
                else
                {
                    continue;
                }
            }
            pHood[ptCt] = x;
            ptCt++;
        }
        return ptCt;
    }

    int32_t GridGeometry1D::RandomHoodI(const int32_t *pHood,
                                        int32_t        hoodLength,
                                        int32_t        centerX,
                                        Rand          &rand) const
    {
        int32_t i = rand.Int(hoodLength / 2);
        return GetHoodI(pHood, hoodLength, centerX, i);
    }

    int32_t GridGeometry1D::GetHoodI(const int32_t *pHood,
                                     int32_t        hoodLength,
                                     int32_t        centerX,
                                     int32_t        entryIndex) const
    {
        int32_t i = entryIndex + hoodLength / 2;
        int32_t x = pHood[i] + centerX;
        if (!InDim(x, m_xDim))
        {
            if (m_wrapX)
            {
                x = Wrap(x, m_xDim);
            }
            else
            {
                return -1;
            }
        }
        return x;
    }
}
