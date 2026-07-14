#include "Hal/Grids/GridGeometry2D.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace Hal
{
    GridGeometry2D::GridGeometry2D(int32_t xDim,
                                   int32_t yDim,
                                   bool    wrapX,
                                   bool    wrapY)
        : m_xDim(xDim),
          m_yDim(yDim),
          m_length(xDim * yDim),
          m_wrapX(wrapX),
          m_wrapY(wrapY)
    {
    }

    GridGeometry2D::~GridGeometry2D()
    {
    }

    int32_t GridGeometry2D::Xdim() const
    {
        return m_xDim;
    }

    int32_t GridGeometry2D::Ydim() const
    {
        return m_yDim;
    }

    int32_t GridGeometry2D::Length() const
    {
        return m_length;
    }

    bool GridGeometry2D::IsWrapX() const
    {
        return m_wrapX;
    }

    bool GridGeometry2D::IsWrapY() const
    {
        return m_wrapY;
    }

    int32_t GridGeometry2D::I(int32_t x,
                              int32_t y) const
    {
        return x * m_yDim + y;
    }

    int32_t GridGeometry2D::I(double x,
                              double y) const
    {
        return static_cast<int32_t>(std::floor(x)) * m_yDim + static_cast<int32_t>(std::floor(y));
    }

    int32_t GridGeometry2D::WrapI(int32_t x,
                                  int32_t y) const
    {
        if (In(x, y))
        {
            return I(x, y);
        }
        if (m_wrapX)
        {
            x = Wrap(x, m_xDim);
        }
        // NOTE: HAL's Java WrapI wrapped y using Xdim() instead of Ydim()
        // (a bug, see docs/JAVA_BUGS_FOUND.md #4); fixed here.
        if (m_wrapY)
        {
            y = Wrap(y, m_yDim);
        }
        if (In(x, y))
        {
            return I(x, y);
        }
        throw std::invalid_argument("cannot map to index in bounds");
    }

    double GridGeometry2D::WrapX(double x) const
    {
        return Wrap(x, static_cast<double>(m_xDim));
    }

    double GridGeometry2D::WrapY(double y) const
    {
        return Wrap(y, static_cast<double>(m_yDim));
    }

    int32_t GridGeometry2D::WrapX(int32_t x) const
    {
        return Wrap(x, m_xDim);
    }

    int32_t GridGeometry2D::WrapY(int32_t y) const
    {
        return Wrap(y, m_yDim);
    }

    int32_t GridGeometry2D::ItoX(int32_t i) const
    {
        return i / m_yDim;
    }

    int32_t GridGeometry2D::ItoY(int32_t i) const
    {
        return i % m_yDim;
    }

    bool GridGeometry2D::In(int32_t x,
                            int32_t y) const
    {
        return (x >= 0) && (x < m_xDim) && (y >= 0) && (y < m_yDim);
    }

    bool GridGeometry2D::In(double x,
                            double y) const
    {
        return In(static_cast<int32_t>(std::floor(x)), static_cast<int32_t>(std::floor(y)));
    }

    bool GridGeometry2D::InWrap(int32_t x,
                                int32_t y) const
    {
        return (m_wrapX || InDim(x, m_xDim)) && (m_wrapY || InDim(y, m_yDim));
    }

    double GridGeometry2D::DispX(double x1,
                                 double x2) const
    {
        // NOTE: an earlier version of this port called
        // DispWrap(x2, x1, m_xDim) here, which returns x1 - x2 (adjusted)
        // rather than x2 - x1 (adjusted) — a sign flip relative to the
        // non-wrap branch below, introduced in translation (Java's
        // Agent2DBase.DispX is a member method called as DispX(self, other)
        // and is self-consistent between its wrap/non-wrap branches; this
        // free-function form needs DispWrap(x1, x2, ...) to preserve that).
        // Not a Java bug — a port bug, fixed directly rather than filed in
        // JAVA_BUGS_FOUND.md.
        if (m_wrapX)
        {
            return DispWrap(x1, x2, m_xDim);
        }
        return x2 - x1;
    }

    double GridGeometry2D::DispY(double y1,
                                 double y2) const
    {
        if (m_wrapY)
        {
            return DispWrap(y1, y2, m_yDim);
        }
        return y2 - y1;
    }

    double GridGeometry2D::Dist(double x1,
                                double y1,
                                double x2,
                                double y2) const
    {
        return Norm(DispX(x1, x2), DispY(y1, y2));
    }

    double GridGeometry2D::Dist(int32_t i1,
                                int32_t i2) const
    {
        return Dist(ItoX(i1), ItoY(i1), ItoX(i2), ItoY(i2));
    }

    double GridGeometry2D::DistSquared(double x1,
                                       double y1,
                                       double x2,
                                       double y2) const
    {
        return NormSquared(DispX(x1, x2), DispY(y1, y2));
    }

    int32_t GridGeometry2D::MapXsq(int32_t               x,
                                   const GridGeometry2D &other) const
    {
        return static_cast<int32_t>(((x + 0.5) * other.Xdim()) / m_xDim);
    }

    int32_t GridGeometry2D::MapYsq(int32_t               y,
                                   const GridGeometry2D &other) const
    {
        return static_cast<int32_t>(((y + 0.5) * other.Ydim()) / m_yDim);
    }

    int32_t GridGeometry2D::MapI(int32_t               i,
                                 const GridGeometry2D &other) const
    {
        int32_t x = ItoX(i);
        int32_t y = ItoY(i);
        return other.I(MapXsq(x, other), MapYsq(y, other));
    }

    double GridGeometry2D::MapXpt(double                x,
                                  const GridGeometry2D &other) const
    {
        return x * other.Xdim() / m_xDim;
    }

    double GridGeometry2D::MapYpt(double                y,
                                  const GridGeometry2D &other) const
    {
        return y * other.Ydim() / m_yDim;
    }

    std::vector<int32_t> GridGeometry2D::BoundaryIs() const
    {
        std::vector<int32_t> ret((m_xDim + m_yDim) * 2);
        for (int32_t x = 0; x < m_xDim; x++)
        {
            ret[x] = I(x, 0);
            ret[x + m_xDim] = I(x, m_yDim - 1);
        }
        for (int32_t y = 0; y < m_yDim; y++)
        {
            ret[y + m_xDim * 2] = I(0, y);
            ret[y + m_xDim * 2 + m_yDim] = I(m_xDim - 1, y);
        }
        return ret;
    }

    int32_t GridGeometry2D::AlongLineIs(double   x1,
                                        double   y1,
                                        double   x2,
                                        double   y2,
                                        int32_t *pWriteHere) const
    {
        double dx = std::abs(x2 - x1);
        double dy = std::abs(y2 - y1);

        int32_t x = static_cast<int32_t>(std::floor(x1));
        int32_t y = static_cast<int32_t>(std::floor(y1));

        int32_t n = 1;
        int32_t xInc;
        int32_t yInc;
        double error;

        if (dx == 0)
        {
            xInc = 0;
            error = std::numeric_limits<double>::max();
        }
        else if (x2 > x1)
        {
            xInc = 1;
            n += static_cast<int32_t>(std::floor(x2)) - x;
            error = (std::floor(x1) + 1 - x1) * dy;
        }
        else
        {
            xInc = -1;
            n += x - static_cast<int32_t>(std::floor(x2));
            error = (x1 - std::floor(x1)) * dy;
        }

        if (dy == 0)
        {
            yInc = 0;
            error -= std::numeric_limits<double>::max();
        }
        else if (y2 > y1)
        {
            yInc = 1;
            n += static_cast<int32_t>(std::floor(y2)) - y;
            error -= (std::floor(y1) + 1 - y1) * dx;
        }
        else
        {
            yInc = -1;
            n += y - static_cast<int32_t>(std::floor(y2));
            error -= (y1 - std::floor(y1)) * dx;
        }

        int32_t count = 0;
        for (; n > 0; --n)
        {
            pWriteHere[count] = I(static_cast<int32_t>(std::floor(x)), static_cast<int32_t>(std::floor(y)));
            count++;

            if (error > 0)
            {
                y += yInc;
                error -= dx;
            }
            else
            {
                x += xInc;
                error += dy;
            }
        }
        return count;
    }

    int32_t GridGeometry2D::GetHoodI(const int32_t *pHood,
                                     int32_t        hoodLength,
                                     int32_t        centerX,
                                     int32_t        centerY,
                                     int32_t        entryIndex) const
    {
        int32_t i = entryIndex * 2 + hoodLength / 3;
        int32_t x = pHood[i] + centerX;
        int32_t y = pHood[i + 1] + centerY;
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
        if (!InDim(y, m_yDim))
        {
            if (m_wrapY)
            {
                y = Wrap(y, m_yDim);
            }
            else
            {
                return -1;
            }
        }
        return I(x, y);
    }

    int32_t GridGeometry2D::RandomHoodI(const int32_t *pHood,
                                        int32_t        hoodLength,
                                        int32_t        centerX,
                                        int32_t        centerY,
                                        Rand          &rand) const
    {
        int32_t i = rand.Int(hoodLength / 3);
        return GetHoodI(pHood, hoodLength, centerX, centerY, i);
    }

    int32_t GridGeometry2D::MapHood(int32_t *pHood,
                                    int32_t  hoodLength,
                                    int32_t  centerX,
                                    int32_t  centerY) const
    {
        int32_t ptCt = 0;
        int32_t iStart = hoodLength / 3;
        for (int32_t i = iStart; i < hoodLength; i += 2)
        {
            int32_t x = pHood[i] + centerX;
            int32_t y = pHood[i + 1] + centerY;
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
            if (!InDim(y, m_yDim))
            {
                if (m_wrapY)
                {
                    y = Wrap(y, m_yDim);
                }
                else
                {
                    continue;
                }
            }
            pHood[ptCt] = I(x, y);
            ptCt++;
        }
        return ptCt;
    }

    int32_t GridGeometry2D::MapHood(int32_t *pHood,
                                    int32_t  hoodLength,
                                    int32_t  centerI) const
    {
        return MapHood(pHood, hoodLength, ItoX(centerI), ItoY(centerI));
    }
}
