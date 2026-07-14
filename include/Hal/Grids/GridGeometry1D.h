#pragma once

#include "Hal/MathUtil.h"
#include "Hal/Rand.h"

#include <cstdint>

namespace Hal
{
    // Shared indexing/neighborhood geometry for 1D grids, ported from
    // HAL's Interfaces/Grid1D.java. Unlike the Java version (a default-
    // method interface implemented by each concrete grid class via virtual
    // dispatch), this is a plain concrete base class: xDim/wrapX are held
    // directly as members here rather than behind virtual accessors, since
    // every 1D grid type (value grid, agent grid, PDE grid) stores exactly
    // the same dimension/wrap state. This keeps the neighbor-search hot
    // path (Phase 2's agent cell-list lookups) free of vtable dispatch.
    //
    // Neighborhood-taking methods (ApplyHood/MapHood/etc.) are templates on
    // the caller's callable type rather than std::function, so they accept
    // captured lambdas/functors with zero type-erasure overhead — the
    // template plays the role Java's per-call lambda + functional-interface
    // played, without needing a named struct at every call site.
    class GridGeometry1D
    {
    public:
        GridGeometry1D(int32_t xDim,
                       bool    wrapX);

        ~GridGeometry1D();

        int32_t Xdim() const;
        int32_t Length() const;
        bool IsWrapX() const;

        // Wraps x into a valid index, or throws if out of bounds and not
        // wrapped.
        int32_t WrapI(int32_t x) const;

        double WrapX(double x) const;
        int32_t WrapX(int32_t x) const;

        bool In(int32_t x) const;
        bool In(double x) const;
        bool InWrap(int32_t x) const;
        bool InWrap(double x) const;

        double DispX(double x1,
                     double x2) const;

        double Dist(double x1,
                    double x2) const;

        double Dist(int32_t x1,
                    int32_t x2) const;

        double DistSquared(double x1,
                           double x2) const;

        // Returns the index of the center of the square in `other` that x
        // maps to, when rescaling between grids of different dimensions.
        int32_t MapXsq(int32_t               x,
                       const GridGeometry1D &other) const;

        double MapXpt(double                x,
                      const GridGeometry1D &other) const;

        template <typename ActionFn>
        void ApplyRectangle(int32_t  startX,
                            int32_t  width,
                            ActionFn action) const
        {
            for (int32_t x = startX; x < (startX + width); x++)
            {
                int32_t xFinal = x;
                if (m_wrapX)
                {
                    xFinal = Wrap(x, m_xDim);
                }
                action(xFinal);
            }
        }

        template <typename ActionFn>
        void ApplyHood(int32_t  *pHood,
                       int32_t   hoodLength,
                       int32_t   centerX,
                       ActionFn  action) const
        {
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
                action(x);
            }
        }

        template <typename ActionFn>
        void ApplyHoodWithIndex(int32_t  *pHood,
                                int32_t   hoodLength,
                                int32_t   centerX,
                                ActionFn  action) const
        {
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
                action(i - iStart, x);
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

        // Translates pHood (centered on the origin) to be centered on
        // centerX, writing valid mapped indices back into the front of
        // pHood. Returns the number of valid entries written.
        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        int32_t  centerX) const;

        template <typename EvalFn>
        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        int32_t  centerX,
                        EvalFn   eval) const
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
                if (eval(x))
                {
                    pHood[ptCt] = x;
                    ptCt++;
                }
            }
            return ptCt;
        }

        template <typename IsValidFn>
        bool ContainsValidI(int32_t   *pHood,
                            int32_t    hoodLength,
                            int32_t    centerX,
                            IsValidFn  isValid) const
        {
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
                if (isValid(x))
                {
                    return true;
                }
            }
            return false;
        }

        // Returns a random mapped index from the full neighborhood, or -1
        // if that entry does not map to a valid location.
        int32_t RandomHoodI(const int32_t *pHood,
                            int32_t        hoodLength,
                            int32_t        centerX,
                            Rand          &rand) const;

        int32_t GetHoodI(const int32_t *pHood,
                         int32_t        hoodLength,
                         int32_t        centerX,
                         int32_t        entryIndex) const;

        template <typename ActionFn>
        void ForEdge(ActionFn action) const
        {
            action(0);
            action(m_xDim - 1);
        }

    private:
        int32_t m_xDim;
        int32_t m_length;
        bool m_wrapX;
    };
}
