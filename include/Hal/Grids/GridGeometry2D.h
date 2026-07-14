#pragma once

#include "Hal/MathUtil.h"
#include "Hal/Rand.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    // Shared indexing/neighborhood geometry for 2D grids, ported from
    // HAL's Interfaces/Grid2D.java. See GridGeometry1D.h for the rationale
    // behind using a concrete (non-virtual) base class and templated
    // neighborhood-hook methods instead of Java's default-method interface.
    class GridGeometry2D
    {
    public:
        GridGeometry2D(int32_t xDim,
                       int32_t yDim,
                       bool    wrapX,
                       bool    wrapY);

        ~GridGeometry2D();

        int32_t Xdim() const;
        int32_t Ydim() const;
        int32_t Length() const;
        bool IsWrapX() const;
        bool IsWrapY() const;

        int32_t I(int32_t x,
                  int32_t y) const;

        int32_t I(double x,
                  double y) const;

        int32_t WrapI(int32_t x,
                      int32_t y) const;

        double WrapX(double x) const;
        double WrapY(double y) const;
        int32_t WrapX(int32_t x) const;
        int32_t WrapY(int32_t y) const;

        int32_t ItoX(int32_t i) const;
        int32_t ItoY(int32_t i) const;

        bool In(int32_t x,
                int32_t y) const;

        bool In(double x,
                double y) const;

        bool InWrap(int32_t x,
                    int32_t y) const;

        double DispX(double x1,
                     double x2) const;

        double DispY(double y1,
                     double y2) const;

        double Dist(double x1,
                    double y1,
                    double x2,
                    double y2) const;

        double Dist(int32_t i1,
                    int32_t i2) const;

        double DistSquared(double x1,
                           double y1,
                           double x2,
                           double y2) const;

        int32_t MapXsq(int32_t               x,
                       const GridGeometry2D &other) const;

        int32_t MapYsq(int32_t               y,
                       const GridGeometry2D &other) const;

        int32_t MapI(int32_t               i,
                     const GridGeometry2D &other) const;

        double MapXpt(double                x,
                      const GridGeometry2D &other) const;

        double MapYpt(double                y,
                      const GridGeometry2D &other) const;

        // Returns one index per boundary square, in [bottom row, top row,
        // left column, right column] order.
        std::vector<int32_t> BoundaryIs() const;

        // Along-line rasterization (Bresenham-like): writes the indices of
        // every square the line from (x1,y1) to (x2,y2) touches into
        // pWriteHere, returning the count written.
        int32_t AlongLineIs(double   x1,
                            double   y1,
                            double   x2,
                            double   y2,
                            int32_t *pWriteHere) const;

        int32_t GetHoodI(const int32_t *pHood,
                         int32_t        hoodLength,
                         int32_t        centerX,
                         int32_t        centerY,
                         int32_t        entryIndex) const;

        int32_t RandomHoodI(const int32_t *pHood,
                            int32_t        hoodLength,
                            int32_t        centerX,
                            int32_t        centerY,
                            Rand          &rand) const;

        template <typename ActionFn>
        void ApplyRectangle(int32_t  startX,
                            int32_t  startY,
                            int32_t  width,
                            int32_t  height,
                            ActionFn action) const
        {
            for (int32_t x = startX; x < (startX + width); x++)
            {
                for (int32_t y = startY; y < (startY + height); y++)
                {
                    int32_t xFinal = x;
                    int32_t yFinal = y;
                    if (m_wrapX)
                    {
                        xFinal = Wrap(x, m_xDim);
                    }
                    if (m_wrapY)
                    {
                        yFinal = Wrap(y, m_yDim);
                    }
                    action(xFinal, yFinal);
                }
            }
        }

        template <typename ActionFn>
        void ApplyHoodMapped(int32_t  *pHood,
                             int32_t   validCount,
                             ActionFn  action) const
        {
            for (int32_t i = 0; i < validCount; i++)
            {
                action(pHood[i]);
            }
        }

        template <typename ActionFn>
        void ApplyHood(const int32_t *pHood,
                       int32_t        hoodLength,
                       int32_t        centerX,
                       int32_t        centerY,
                       ActionFn       action) const
        {
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
                action(x, y);
            }
        }

        template <typename ActionFn>
        void ApplyHood(const int32_t *pHood,
                       int32_t        hoodLength,
                       int32_t        centerI,
                       ActionFn       action) const
        {
            ApplyHood(pHood, hoodLength, ItoX(centerI), ItoY(centerI), action);
        }

        template <typename ActionFn>
        void ApplyHoodWithIndex(const int32_t *pHood,
                                int32_t        hoodLength,
                                int32_t        centerX,
                                int32_t        centerY,
                                ActionFn       action) const
        {
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
                action((i - iStart) / 2, x, y);
            }
        }

        // Translates pHood (centered on the origin) to be centered on
        // (centerX, centerY), writing mapped indices back into the front
        // of pHood. Returns the number of valid entries written.
        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        int32_t  centerX,
                        int32_t  centerY) const;

        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        int32_t  centerI) const;

        template <typename EvalFn>
        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        int32_t  centerX,
                        int32_t  centerY,
                        EvalFn   eval) const
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
                int32_t j = I(x, y);
                if (eval(j, x, y))
                {
                    pHood[ptCt] = j;
                    ptCt++;
                }
            }
            return ptCt;
        }

        template <typename EvalFn>
        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        int32_t  centerI,
                        EvalFn   eval) const
        {
            return MapHood(pHood, hoodLength, ItoX(centerI), ItoY(centerI), eval);
        }

        template <typename IsValidFn>
        bool ContainsValidI(const int32_t *pHood,
                            int32_t        hoodLength,
                            int32_t        centerX,
                            int32_t        centerY,
                            IsValidFn      isValid) const
        {
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
                if (isValid(x, y))
                {
                    return true;
                }
            }
            return false;
        }

        template <typename ActionFn>
        void ForEdge(ActionFn action) const
        {
            for (int32_t x = 0; x < m_xDim; x++)
            {
                int32_t y = 0;
                action(I(x, y), x, y);
                y = m_yDim - 1;
                action(I(x, y), x, y);
            }
            for (int32_t y = 1; y < (m_yDim - 1); y++)
            {
                int32_t x = 0;
                action(I(x, y), x, y);
                x = m_xDim - 1;
                action(I(x, y), x, y);
            }
        }

    private:
        int32_t m_xDim;
        int32_t m_yDim;
        int32_t m_length;
        bool m_wrapX;
        bool m_wrapY;
    };
}
