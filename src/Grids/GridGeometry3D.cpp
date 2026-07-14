#include "Hal/Grids/GridGeometry3D.h"

#include <cmath>
#include <stdexcept>

namespace Hal
{
    GridGeometry3D::GridGeometry3D(int32_t xDim,
                                   int32_t yDim,
                                   int32_t zDim,
                                   bool    wrapX,
                                   bool    wrapY,
                                   bool    wrapZ)
        : m_xDim(xDim),
          m_yDim(yDim),
          m_zDim(zDim),
          m_length(xDim * yDim * zDim),
          m_wrapX(wrapX),
          m_wrapY(wrapY),
          m_wrapZ(wrapZ)
    {
    }

    GridGeometry3D::~GridGeometry3D()
    {
    }

    int32_t GridGeometry3D::Xdim() const
    {
        return m_xDim;
    }

    int32_t GridGeometry3D::Ydim() const
    {
        return m_yDim;
    }

    int32_t GridGeometry3D::Zdim() const
    {
        return m_zDim;
    }

    int32_t GridGeometry3D::Length() const
    {
        return m_length;
    }

    bool GridGeometry3D::IsWrapX() const
    {
        return m_wrapX;
    }

    bool GridGeometry3D::IsWrapY() const
    {
        return m_wrapY;
    }

    bool GridGeometry3D::IsWrapZ() const
    {
        return m_wrapZ;
    }

    int32_t GridGeometry3D::I(int32_t x,
                              int32_t y,
                              int32_t z) const
    {
        return x * m_yDim * m_zDim + y * m_zDim + z;
    }

    int32_t GridGeometry3D::I(double x,
                              double y,
                              double z) const
    {
        return static_cast<int32_t>(std::floor(x)) * m_yDim * m_zDim +
               static_cast<int32_t>(std::floor(y)) * m_zDim +
               static_cast<int32_t>(std::floor(z));
    }

    int32_t GridGeometry3D::WrapI(int32_t x,
                                  int32_t y,
                                  int32_t z) const
    {
        if (In(x, y, z))
        {
            return I(x, y, z);
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
        if (m_wrapZ)
        {
            z = Wrap(z, m_zDim);
        }
        if (In(x, y, z))
        {
            return I(x, y, z);
        }
        throw std::invalid_argument("cannot map to index in bounds");
    }

    double GridGeometry3D::WrapX(double x) const
    {
        return Wrap(x, static_cast<double>(m_xDim));
    }

    double GridGeometry3D::WrapY(double y) const
    {
        return Wrap(y, static_cast<double>(m_yDim));
    }

    double GridGeometry3D::WrapZ(double z) const
    {
        return Wrap(z, static_cast<double>(m_zDim));
    }

    int32_t GridGeometry3D::WrapX(int32_t x) const
    {
        return Wrap(x, m_xDim);
    }

    int32_t GridGeometry3D::WrapY(int32_t y) const
    {
        return Wrap(y, m_yDim);
    }

    int32_t GridGeometry3D::WrapZ(int32_t z) const
    {
        return Wrap(z, m_zDim);
    }

    int32_t GridGeometry3D::ItoX(int32_t i) const
    {
        return i / (m_yDim * m_zDim);
    }

    int32_t GridGeometry3D::ItoY(int32_t i) const
    {
        return (i / m_zDim) % m_yDim;
    }

    int32_t GridGeometry3D::ItoZ(int32_t i) const
    {
        return i % m_zDim;
    }

    bool GridGeometry3D::In(int32_t x,
                            int32_t y,
                            int32_t z) const
    {
        return (x >= 0) && (x < m_xDim) && (y >= 0) && (y < m_yDim) && (z >= 0) && (z < m_zDim);
    }

    bool GridGeometry3D::In(double x,
                            double y,
                            double z) const
    {
        return In(static_cast<int32_t>(std::floor(x)), static_cast<int32_t>(std::floor(y)), static_cast<int32_t>(std::floor(z)));
    }

    bool GridGeometry3D::InWrap(int32_t x,
                                int32_t y,
                                int32_t z) const
    {
        return (m_wrapX || InDim(x, m_xDim)) && (m_wrapY || InDim(y, m_yDim)) && (m_wrapZ || InDim(z, m_zDim));
    }

    bool GridGeometry3D::InWrap(double x,
                                double y,
                                double z) const
    {
        return (m_wrapX || InDim(x, static_cast<double>(m_xDim))) &&
               (m_wrapY || InDim(y, static_cast<double>(m_yDim))) &&
               (m_wrapZ || InDim(z, static_cast<double>(m_zDim)));
    }

    double GridGeometry3D::DispX(double x1,
                                 double x2) const
    {
        // NOTE: same sign-flip port bug as GridGeometry2D::DispX, fixed the
        // same way — see the comment there.
        if (m_wrapX)
        {
            return DispWrap(x1, x2, m_xDim);
        }
        return x2 - x1;
    }

    double GridGeometry3D::DispY(double y1,
                                 double y2) const
    {
        if (m_wrapY)
        {
            return DispWrap(y1, y2, m_yDim);
        }
        return y2 - y1;
    }

    double GridGeometry3D::DispZ(double z1,
                                 double z2) const
    {
        // NOTE: HAL's Java DispZ checked IsWrapY() instead of IsWrapZ()
        // (a bug, see docs/JAVA_BUGS_FOUND.md #5); fixed here.
        if (m_wrapZ)
        {
            return DispWrap(z1, z2, m_zDim);
        }
        return z2 - z1;
    }

    double GridGeometry3D::Dist(double x1,
                                double y1,
                                double z1,
                                double x2,
                                double y2,
                                double z2) const
    {
        return std::sqrt(DistSquared(x1, y1, z1, x2, y2, z2));
    }

    double GridGeometry3D::Dist(int32_t i1,
                                int32_t i2) const
    {
        return Dist(ItoX(i1), ItoY(i1), ItoZ(i1), ItoX(i2), ItoY(i2), ItoZ(i2));
    }

    double GridGeometry3D::Dist(int32_t i1,
                                int32_t x2,
                                int32_t y2,
                                int32_t z2) const
    {
        return Dist(ItoX(i1), ItoY(i1), ItoZ(i1), x2, y2, z2);
    }

    double GridGeometry3D::DistSquared(double x1,
                                       double y1,
                                       double z1,
                                       double x2,
                                       double y2,
                                       double z2) const
    {
        return NormSquared(DispX(x1, x2), DispY(y1, y2), DispZ(z1, z2));
    }

    int32_t GridGeometry3D::ConvXsq(int32_t               x,
                                    const GridGeometry3D &other) const
    {
        return static_cast<int32_t>(((x + 0.5) * other.Xdim()) / m_xDim);
    }

    int32_t GridGeometry3D::ConvYsq(int32_t               y,
                                    const GridGeometry3D &other) const
    {
        return static_cast<int32_t>(((y + 0.5) * other.Ydim()) / m_yDim);
    }

    int32_t GridGeometry3D::ConvZsq(int32_t               z,
                                    const GridGeometry3D &other) const
    {
        return static_cast<int32_t>(((z + 0.5) * other.Zdim()) / m_zDim);
    }

    int32_t GridGeometry3D::ConvI(int32_t               i,
                                  const GridGeometry3D &other) const
    {
        int32_t x = ItoX(i);
        int32_t y = ItoY(i);
        int32_t z = ItoZ(i);
        return other.I(ConvXsq(x, other), ConvYsq(y, other), ConvZsq(z, other));
    }

    double GridGeometry3D::ConvXpt(double                x,
                                   const GridGeometry3D &other) const
    {
        return x * other.Xdim() / m_xDim;
    }

    double GridGeometry3D::ConvYpt(double                y,
                                   const GridGeometry3D &other) const
    {
        return y * other.Ydim() / m_yDim;
    }

    double GridGeometry3D::ConvZpt(double                z,
                                   const GridGeometry3D &other) const
    {
        return z * other.Zdim() / m_zDim;
    }

    std::vector<int32_t> GridGeometry3D::BoundaryIs() const
    {
        std::vector<int32_t> ret((m_xDim * m_yDim + m_xDim * m_zDim + m_yDim * m_zDim) * 2);
        int32_t side1 = m_xDim * m_yDim;
        int32_t side2 = m_xDim * m_zDim;

        for (int32_t x = 0; x < m_xDim; x++)
        {
            for (int32_t y = 0; y < m_yDim; y++)
            {
                ret[x * m_yDim + y] = I(x, y, 0);
                ret[x * m_yDim + y + side1] = I(x, y, m_zDim - 1);
            }
        }
        for (int32_t x = 0; x < m_xDim; x++)
        {
            for (int32_t z = 0; z < m_zDim; z++)
            {
                ret[x * m_zDim + z + side1 * 2] = I(x, 0, z);
                ret[x * m_zDim + z + side1 * 2 + side2] = I(x, m_yDim - 1, z);
            }
        }
        for (int32_t y = 0; y < m_yDim; y++)
        {
            for (int32_t z = 0; z < m_zDim; z++)
            {
                ret[y * m_zDim + z + (side1 + side2) * 2] = I(0, y, z);
                ret[y * m_zDim + z + (side1 + side2) * 2 + m_yDim * m_zDim] = I(m_xDim - 1, y, z);
            }
        }
        return ret;
    }

    int32_t GridGeometry3D::GetHoodI(const int32_t *pHood,
                                     int32_t        hoodLength,
                                     int32_t        centerX,
                                     int32_t        centerY,
                                     int32_t        centerZ,
                                     int32_t        entryIndex) const
    {
        int32_t i = entryIndex * 3 + hoodLength / 4;
        int32_t x = pHood[i] + centerX;
        int32_t y = pHood[i + 1] + centerY;
        int32_t z = pHood[i + 2] + centerZ;
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
        if (!InDim(z, m_zDim))
        {
            // NOTE: HAL's Java GetHoodI assigned the wrapped z value into y
            // instead of z here (a bug, see docs/JAVA_BUGS_FOUND.md #6),
            // corrupting y and leaving z unwrapped; fixed here.
            if (m_wrapZ)
            {
                z = Wrap(z, m_zDim);
            }
            else
            {
                return -1;
            }
        }
        return I(x, y, z);
    }

    int32_t GridGeometry3D::RandomHoodI(const int32_t *pHood,
                                        int32_t        hoodLength,
                                        int32_t        centerX,
                                        int32_t        centerY,
                                        int32_t        centerZ,
                                        Rand          &rand) const
    {
        int32_t i = rand.Int(hoodLength / 4);
        return GetHoodI(pHood, hoodLength, centerX, centerY, centerZ, i);
    }

    int32_t GridGeometry3D::MapHood(int32_t *pHood,
                                    int32_t  hoodLength,
                                    int32_t  centerX,
                                    int32_t  centerY,
                                    int32_t  centerZ) const
    {
        int32_t ptCt = 0;
        int32_t iStart = hoodLength / 4;
        for (int32_t i = iStart; i < hoodLength; i += 3)
        {
            int32_t x = pHood[i] + centerX;
            int32_t y = pHood[i + 1] + centerY;
            int32_t z = pHood[i + 2] + centerZ;
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
            if (!InDim(z, m_zDim))
            {
                if (m_wrapZ)
                {
                    z = Wrap(z, m_zDim);
                }
                else
                {
                    continue;
                }
            }
            pHood[ptCt] = I(x, y, z);
            ptCt++;
        }
        return ptCt;
    }

    int32_t GridGeometry3D::MapHood(int32_t *pHood,
                                    int32_t  hoodLength,
                                    int32_t  centerI) const
    {
        return MapHood(pHood, hoodLength, ItoX(centerI), ItoY(centerI), ItoZ(centerI));
    }
}
