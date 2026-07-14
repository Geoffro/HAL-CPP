#pragma once

#include "Hal/MathUtil.h"
#include "Hal/Rand.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace Hal
{
    // Shared indexing/neighborhood geometry for 3D grids, ported from
    // HAL's Interfaces/Grid3D.java. See GridGeometry1D.h for the rationale
    // behind the concrete-base-class + templated-callback design.
    //
    // `BucketFillMask`/`BorderFindMask` (flood-fill mask helpers) keep
    // Java's caller-owned-scratch contract: the caller supplies the
    // connectivity hood plus the `out`/`scratch`/`stack` buffers (all sized
    // to `Length()`), so the hot path stays allocation-free.
    class GridGeometry3D
    {
    public:
        GridGeometry3D(int32_t xDim,
                       int32_t yDim,
                       int32_t zDim,
                       bool    wrapX,
                       bool    wrapY,
                       bool    wrapZ);

        ~GridGeometry3D();

        int32_t Xdim() const;
        int32_t Ydim() const;
        int32_t Zdim() const;
        int32_t Length() const;
        bool IsWrapX() const;
        bool IsWrapY() const;
        bool IsWrapZ() const;

        int32_t I(int32_t x,
                  int32_t y,
                  int32_t z) const;

        int32_t I(double x,
                  double y,
                  double z) const;

        int32_t WrapI(int32_t x,
                      int32_t y,
                      int32_t z) const;

        double WrapX(double x) const;
        double WrapY(double y) const;
        double WrapZ(double z) const;
        int32_t WrapX(int32_t x) const;
        int32_t WrapY(int32_t y) const;
        int32_t WrapZ(int32_t z) const;

        int32_t ItoX(int32_t i) const;
        int32_t ItoY(int32_t i) const;
        int32_t ItoZ(int32_t i) const;

        bool In(int32_t x,
                int32_t y,
                int32_t z) const;

        bool In(double x,
                double y,
                double z) const;

        bool InWrap(int32_t x,
                    int32_t y,
                    int32_t z) const;

        bool InWrap(double x,
                    double y,
                    double z) const;

        double DispX(double x1,
                     double x2) const;

        double DispY(double y1,
                     double y2) const;

        double DispZ(double z1,
                     double z2) const;

        double Dist(double x1,
                    double y1,
                    double z1,
                    double x2,
                    double y2,
                    double z2) const;

        double Dist(int32_t i1,
                    int32_t i2) const;

        double Dist(int32_t i1,
                    int32_t x2,
                    int32_t y2,
                    int32_t z2) const;

        double DistSquared(double x1,
                           double y1,
                           double z1,
                           double x2,
                           double y2,
                           double z2) const;

        int32_t ConvXsq(int32_t               x,
                        const GridGeometry3D &other) const;

        int32_t ConvYsq(int32_t               y,
                        const GridGeometry3D &other) const;

        int32_t ConvZsq(int32_t               z,
                        const GridGeometry3D &other) const;

        int32_t ConvI(int32_t               i,
                      const GridGeometry3D &other) const;

        double ConvXpt(double                x,
                       const GridGeometry3D &other) const;

        double ConvYpt(double                y,
                       const GridGeometry3D &other) const;

        double ConvZpt(double                z,
                       const GridGeometry3D &other) const;

        std::vector<int32_t> BoundaryIs() const;

        int32_t GetHoodI(const int32_t *pHood,
                         int32_t        hoodLength,
                         int32_t        centerX,
                         int32_t        centerY,
                         int32_t        centerZ,
                         int32_t        entryIndex) const;

        int32_t RandomHoodI(const int32_t *pHood,
                            int32_t        hoodLength,
                            int32_t        centerX,
                            int32_t        centerY,
                            int32_t        centerZ,
                            Rand          &rand) const;

        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        int32_t  centerX,
                        int32_t  centerY,
                        int32_t  centerZ) const;

        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        int32_t  centerI) const;

        template <typename ActionFn>
        void ApplyRectangle(int32_t  startX,
                            int32_t  startY,
                            int32_t  startZ,
                            int32_t  width,
                            int32_t  height,
                            int32_t  depth,
                            ActionFn action) const
        {
            for (int32_t x = startX; x < (startX + width); x++)
            {
                for (int32_t y = startY; y < (startY + height); y++)
                {
                    for (int32_t z = startZ; z < (startZ + depth); z++)
                    {
                        int32_t xFinal = x;
                        int32_t yFinal = y;
                        int32_t zFinal = z;
                        if (m_wrapX)
                        {
                            xFinal = Wrap(x, m_xDim);
                        }
                        if (m_wrapY)
                        {
                            yFinal = Wrap(y, m_yDim);
                        }
                        if (m_wrapZ)
                        {
                            zFinal = Wrap(z, m_zDim);
                        }
                        action(xFinal, yFinal, zFinal);
                    }
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
                       int32_t        centerZ,
                       ActionFn       action) const
        {
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
                action(x, y, z);
            }
        }

        template <typename ActionFn>
        void ApplyHood(const int32_t *pHood,
                       int32_t        hoodLength,
                       int32_t        centerI,
                       ActionFn       action) const
        {
            ApplyHood(pHood, hoodLength, ItoX(centerI), ItoY(centerI), ItoZ(centerI), action);
        }

        template <typename ActionFn>
        void ApplyHoodWithIndex(const int32_t *pHood,
                                int32_t        hoodLength,
                                int32_t        centerX,
                                int32_t        centerY,
                                int32_t        centerZ,
                                ActionFn       action) const
        {
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
                action((i - iStart) / 3, x, y, z);
            }
        }

        template <typename EvalFn>
        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        int32_t  centerX,
                        int32_t  centerY,
                        int32_t  centerZ,
                        EvalFn   eval) const
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
                int32_t j = I(x, y, z);
                if (eval(j, x, y, z))
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
            return MapHood(pHood, hoodLength, ItoX(centerI), ItoY(centerI), ItoZ(centerI), eval);
        }

        template <typename IsValidFn>
        bool ContainsValidI(const int32_t *pHood,
                            int32_t        hoodLength,
                            int32_t        centerX,
                            int32_t        centerY,
                            int32_t        centerZ,
                            IsValidFn      isValid) const
        {
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
                if (isValid(x, y, z))
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
                for (int32_t y = 0; y < m_yDim; y++)
                {
                    int32_t z = 0;
                    action(I(x, y, z), x, y, z);
                    z = m_zDim - 1;
                    action(I(x, y, z), x, y, z);
                }
            }
            for (int32_t x = 0; x < m_xDim; x++)
            {
                for (int32_t z = 1; z < (m_zDim - 1); z++)
                {
                    int32_t y = 0;
                    action(I(x, y, z), x, y, z);
                    y = m_yDim - 1;
                    action(I(x, y, z), x, y, z);
                }
            }
            for (int32_t y = 1; y < (m_yDim - 1); y++)
            {
                for (int32_t z = 1; z < (m_zDim - 1); z++)
                {
                    int32_t x = 0;
                    action(I(x, y, z), x, y, z);
                    x = m_xDim - 1;
                    action(I(x, y, z), x, y, z);
                }
            }
        }

        // Flood-fill from iStart across cells connected through pNearbyHood,
        // writing `true` into pOut for every reachable cell that satisfies
        // isMask (the fill is confined to the masked region). pOut, pScratch
        // and stack are caller-owned scratch sized to Length(); pNearbyHood
        // is the connectivity offset table (reused as MapHood's write buffer).
        // Ported from Grid3D.java BucketFillMask.
        //
        // Faithful quirk (see JAVA_BUGS_FOUND.md #17): the seed cell iStart
        // is NEVER written into pOut, even when it satisfies isMask — only
        // cells discovered as masked *neighbors* are marked. Callers that
        // want the seed included must set pOut[iStart] themselves.
        template <typename IsMaskFn>
        void BucketFillMask(bool                 *pOut,
                            int32_t              *pNearbyHood,
                            int32_t               hoodLength,
                            bool                 *pScratch,
                            std::vector<int32_t> &stack,
                            int32_t               iStart,
                            IsMaskFn              isMask) const
        {
            std::fill(pScratch, pScratch + m_length, false);
            std::fill(pOut, pOut + m_length, false);
            stack.clear();
            stack.push_back(iStart);
            pScratch[iStart] = true;
            while (!stack.empty())
            {
                int32_t i = stack.back();
                stack.pop_back();
                int32_t x = ItoX(i);
                int32_t y = ItoY(i);
                int32_t z = ItoZ(i);
                int32_t ct = MapHood(pNearbyHood, hoodLength, x, y, z,
                                     [&](int32_t i2, int32_t, int32_t, int32_t) { return !pScratch[i2]; });
                for (int32_t j = 0; j < ct; j++)
                {
                    int32_t i3 = pNearbyHood[j];
                    int32_t x3 = ItoX(i3);
                    int32_t y3 = ItoY(i3);
                    int32_t z3 = ItoZ(i3);
                    pScratch[i3] = true;
                    if (isMask(i3, x3, y3, z3))
                    {
                        pOut[i3] = true;
                        stack.push_back(i3);
                    }
                }
            }
        }

        // Flood-fill from iStart across cells *not* in the mask, marking in
        // pOut the masked cells encountered at the fill's frontier — i.e. the
        // border of the masked region facing the open region containing
        // iStart. Same caller-owned-scratch contract as BucketFillMask.
        // Ported from Grid3D.java BorderFindMask.
        template <typename IsMaskFn>
        void BorderFindMask(bool                 *pOut,
                            int32_t              *pNearbyHood,
                            int32_t               hoodLength,
                            bool                 *pScratch,
                            std::vector<int32_t> &stack,
                            int32_t               iStart,
                            IsMaskFn              isMask) const
        {
            std::fill(pScratch, pScratch + m_length, false);
            std::fill(pOut, pOut + m_length, false);
            stack.clear();
            stack.push_back(iStart);
            pScratch[iStart] = true;
            while (!stack.empty())
            {
                int32_t i = stack.back();
                stack.pop_back();
                int32_t x = ItoX(i);
                int32_t y = ItoY(i);
                int32_t z = ItoZ(i);
                if (isMask(i, x, y, z))
                {
                    pOut[i] = true;
                }
                else
                {
                    int32_t ct = MapHood(pNearbyHood, hoodLength, x, y, z,
                                         [&](int32_t i2, int32_t, int32_t, int32_t) { return !pScratch[i2]; });
                    for (int32_t j = 0; j < ct; j++)
                    {
                        int32_t i3 = pNearbyHood[j];
                        pScratch[i3] = true;
                        stack.push_back(i3);
                    }
                }
            }
        }

    private:
        int32_t m_xDim;
        int32_t m_yDim;
        int32_t m_zDim;
        int32_t m_length;
        bool m_wrapX;
        bool m_wrapY;
        bool m_wrapZ;
    };
}
