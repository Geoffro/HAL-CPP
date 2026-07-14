#pragma once

#include "Hal/MathUtil.h"

#include <cstdint>

namespace Hal
{
    namespace Internal
    {
        // Explicit finite-difference diffusion/advection stencils, ported
        // from Tools/Internal/PDEequations.java (the homogeneous-rate
        // overloads only — the discontinuous/spatially-varying-rate
        // overloads are deferred).
        //
        // Each axis has two variants: a no-boundary-condition version
        // (zero-flux/Neumann — contributes nothing across a non-wrapped
        // edge) and a version taking a boundary-condition callable
        // (Dirichlet — reflects the boundary value with a factor of 2).
        // These are genuinely different formulas, not the same formula
        // with a "BC that returns 0" plugged in (see the two branches
        // below), so they're kept as separate functions per
        // docs/DESIGN.md's callback-style note: hot-loop hooks like this
        // are templated on the caller's callable type rather than
        // std::function, for zero-overhead captured lambdas.

        double Delta1D(const double *pField,
                       double        centerVal,
                       int32_t       x,
                       int32_t       xDim,
                       bool          wrapX);

        template <typename BCFn>
        double Delta1D(const double *pField,
                       double        centerVal,
                       int32_t       x,
                       int32_t       xDim,
                       bool          wrapX,
                       BCFn          bc)
        {
            if (InDim(x, xDim))
            {
                return pField[x] - centerVal;
            }
            if (wrapX)
            {
                return pField[Wrap(x, xDim)] - centerVal;
            }
            return (bc(x) - centerVal) * 2;
        }

        double DeltaX2D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX);

        template <typename BCFn>
        double DeltaX2D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        BCFn          bc)
        {
            if (InDim(x, xDim))
            {
                return pField[x * yDim + y] - centerVal;
            }
            if (wrapX)
            {
                return pField[Wrap(x, xDim) * yDim + y] - centerVal;
            }
            return (bc(x, y) - centerVal) * 2;
        }

        double DeltaY2D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapY);

        template <typename BCFn>
        double DeltaY2D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapY,
                        BCFn          bc)
        {
            if (InDim(y, yDim))
            {
                return pField[x * yDim + y] - centerVal;
            }
            if (wrapY)
            {
                return pField[x * yDim + Wrap(y, yDim)] - centerVal;
            }
            return (bc(x, y) - centerVal) * 2;
        }

        double DeltaX3D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       z,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapX);

        template <typename BCFn>
        double DeltaX3D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       z,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapX,
                        BCFn          bc)
        {
            if (InDim(x, xDim))
            {
                return pField[x * yDim * zDim + y * zDim + z] - centerVal;
            }
            if (wrapX)
            {
                return pField[Wrap(x, xDim) * yDim * zDim + y * zDim + z] - centerVal;
            }
            return (bc(x, y, z) - centerVal) * 2;
        }

        double DeltaY3D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       z,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapY);

        template <typename BCFn>
        double DeltaY3D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       z,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapY,
                        BCFn          bc)
        {
            if (InDim(y, yDim))
            {
                return pField[x * yDim * zDim + y * zDim + z] - centerVal;
            }
            if (wrapY)
            {
                return pField[x * yDim * zDim + Wrap(y, yDim) * zDim + z] - centerVal;
            }
            return (bc(x, y, z) - centerVal) * 2;
        }

        double DeltaZ3D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       z,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapZ);

        template <typename BCFn>
        double DeltaZ3D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       z,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapZ,
                        BCFn          bc)
        {
            if (InDim(z, zDim))
            {
                return pField[x * yDim * zDim + y * zDim + z] - centerVal;
            }
            if (wrapZ)
            {
                return pField[x * yDim * zDim + y * zDim + Wrap(z, zDim)] - centerVal;
            }
            return (bc(x, y, z) - centerVal) * 2;
        }

        // --- Diffusion (5/7-point stencil, uniform rate) ---

        void Diffusion1(const double *pField,
                        double       *pDeltas,
                        double        diffRate,
                        int32_t       xDim,
                        bool          wrapX);

        template <typename BCFn>
        void Diffusion1(const double *pField,
                        double       *pDeltas,
                        double        diffRate,
                        int32_t       xDim,
                        bool          wrapX,
                        BCFn          bc)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                double centerVal = pField[x];
                double valSum = Delta1D(pField, centerVal, x + 1, xDim, wrapX, bc);
                valSum += Delta1D(pField, centerVal, x - 1, xDim, wrapX, bc);
                pDeltas[x] += valSum * diffRate;
            }
        }

        void Diffusion2(const double *pField,
                        double       *pDeltas,
                        double        diffRate,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        bool          wrapY);

        // Value from a per-cell array at (x, y) with x possibly off-grid:
        // the in-bounds value, the wrapped value if the x-axis wraps,
        // otherwise 0 (zero-flux). Used to look up per-cell diffusion rates
        // at neighbor faces.
        double DisplacedX2D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       xDim,
                            int32_t       yDim,
                            bool          wrapX);

        double DisplacedY2D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       xDim,
                            int32_t       yDim,
                            bool          wrapY);

        // Explicit diffusion with a spatially-varying (discontinuous)
        // diffusion rate: pDiffRatesX / pDiffRatesY give the per-cell rate
        // for x- and y-direction flux. Zero-flux boundaries (no boundary
        // condition callback). With uniform rate arrays this reduces
        // exactly to the constant-rate Diffusion2 above.
        void Diffusion2(const double *pField,
                        double       *pDeltas,
                        const double *pDiffRatesX,
                        const double *pDiffRatesY,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        bool          wrapY);

        template <typename BCFn>
        void Diffusion2(const double *pField,
                        double       *pDeltas,
                        double        diffRate,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        bool          wrapY,
                        BCFn          bc)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    int32_t i = x * yDim + y;
                    double  centerVal = pField[i];
                    double  deltaSum  = DeltaX2D(pField, centerVal, x + 1, y, xDim, yDim, wrapX, bc);
                    deltaSum += DeltaX2D(pField, centerVal, x - 1, y, xDim, yDim, wrapX, bc);
                    deltaSum += DeltaY2D(pField, centerVal, x, y + 1, xDim, yDim, wrapY, bc);
                    deltaSum += DeltaY2D(pField, centerVal, x, y - 1, xDim, yDim, wrapY, bc);
                    pDeltas[i] += deltaSum * diffRate;
                }
            }
        }

        void Diffusion3(const double *pField,
                        double       *pDeltas,
                        double        diffRate,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapX,
                        bool          wrapY,
                        bool          wrapZ);

        // Per-cell-array lookups at a possibly off-grid neighbor: in-bounds
        // value, wrapped value if that axis wraps, otherwise 0 (zero-flux).
        // The 3D analogues of DisplacedX2D/DisplacedY2D.
        double DisplacedX3D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       z,
                            int32_t       xDim,
                            int32_t       yDim,
                            int32_t       zDim,
                            bool          wrapX);

        double DisplacedY3D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       z,
                            int32_t       xDim,
                            int32_t       yDim,
                            int32_t       zDim,
                            bool          wrapY);

        double DisplacedZ3D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       z,
                            int32_t       xDim,
                            int32_t       yDim,
                            int32_t       zDim,
                            bool          wrapZ);

        // Explicit 3D diffusion with per-cell (discontinuous) rates for each
        // axis. Zero-flux boundaries (no boundary-condition callback). With
        // uniform rate arrays this reduces exactly to the constant-rate
        // Diffusion3 above.
        void Diffusion3(const double *pField,
                        double       *pDeltas,
                        const double *pDiffRatesX,
                        const double *pDiffRatesY,
                        const double *pDiffRatesZ,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapX,
                        bool          wrapY,
                        bool          wrapZ);

        template <typename BCFn>
        void Diffusion3(const double *pField,
                        double       *pDeltas,
                        double        diffRate,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapX,
                        bool          wrapY,
                        bool          wrapZ,
                        BCFn          bc)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    for (int32_t z = 0; z < zDim; z++)
                    {
                        int32_t i = x * yDim * zDim + y * zDim + z;
                        double  centerVal = pField[i];
                        double  deltaSum  = DeltaX3D(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX, bc);
                        deltaSum += DeltaX3D(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX, bc);
                        deltaSum += DeltaY3D(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY, bc);
                        deltaSum += DeltaY3D(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY, bc);
                        deltaSum += DeltaZ3D(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ, bc);
                        deltaSum += DeltaZ3D(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ, bc);
                        pDeltas[i] += deltaSum * diffRate;
                    }
                }
            }
        }

        // --- Advection (first-order upwind, uniform velocity) ---

        void Advection1(const double *pField,
                        double       *pDeltas,
                        double        vel,
                        int32_t       xDim,
                        bool          wrapX);

        template <typename BCFn>
        void Advection1(const double *pField,
                        double       *pDeltas,
                        double        vel,
                        int32_t       xDim,
                        bool          wrapX,
                        BCFn          bc)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                if (vel > 0)
                {
                    pDeltas[x] += Delta1D(pField, pField[x], x - 1, xDim, wrapX, bc) * vel;
                }
                else if (vel < 0)
                {
                    pDeltas[x] -= Delta1D(pField, pField[x], x + 1, xDim, wrapX, bc) * vel;
                }
            }
        }

        void Advection2(const double *pField,
                        double       *pDeltas,
                        double        xVel,
                        double        yVel,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        bool          wrapY);

        template <typename BCFn>
        void Advection2(const double *pField,
                        double       *pDeltas,
                        double        xVel,
                        double        yVel,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        bool          wrapY,
                        BCFn          bc)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    int32_t i = x * yDim + y;
                    double  centerVal = pField[i];
                    if (xVel > 0)
                    {
                        pDeltas[i] += DeltaX2D(pField, centerVal, x - 1, y, xDim, yDim, wrapX, bc) * xVel;
                    }
                    else if (xVel < 0)
                    {
                        pDeltas[i] -= DeltaX2D(pField, centerVal, x + 1, y, xDim, yDim, wrapX, bc) * xVel;
                    }
                    if (yVel > 0)
                    {
                        pDeltas[i] += DeltaY2D(pField, centerVal, x, y - 1, xDim, yDim, wrapY, bc) * yVel;
                    }
                    else if (yVel < 0)
                    {
                        pDeltas[i] -= DeltaY2D(pField, centerVal, x, y + 1, xDim, yDim, wrapY, bc) * yVel;
                    }
                }
            }
        }

        void Advection3(const double *pField,
                        double       *pDeltas,
                        double        xVel,
                        double        yVel,
                        double        zVel,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapX,
                        bool          wrapY,
                        bool          wrapZ);

        template <typename BCFn>
        void Advection3(const double *pField,
                        double       *pDeltas,
                        double        xVel,
                        double        yVel,
                        double        zVel,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapX,
                        bool          wrapY,
                        bool          wrapZ,
                        BCFn          bc)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    for (int32_t z = 0; z < zDim; z++)
                    {
                        int32_t i = x * yDim * zDim + y * zDim + z;
                        double  centerVal = pField[i];
                        if (xVel > 0)
                        {
                            pDeltas[i] += DeltaX3D(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX, bc) * xVel;
                        }
                        else if (xVel < 0)
                        {
                            pDeltas[i] -= DeltaX3D(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX, bc) * xVel;
                        }
                        if (yVel > 0)
                        {
                            pDeltas[i] += DeltaY3D(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY, bc) * yVel;
                        }
                        else if (yVel < 0)
                        {
                            pDeltas[i] -= DeltaY3D(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY, bc) * yVel;
                        }
                        if (zVel > 0)
                        {
                            pDeltas[i] += DeltaZ3D(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ, bc) * zVel;
                        }
                        else if (zVel < 0)
                        {
                            pDeltas[i] -= DeltaZ3D(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ, bc) * zVel;
                        }
                    }
                }
            }
        }

        // --- Advection (first-order upwind, per-cell velocity field) ---
        //
        // The spatially-varying-velocity analogue of the uniform-velocity
        // advection above, ported from Java's per-cell-array Advection1/2/3
        // overloads. Unlike the uniform version's upwind-delta form, these
        // use the conservative flux form (per-face upwind fluxes summed
        // into each cell), matching Java: with a varying velocity the two
        // are no longer equivalent, and the flux form is what conserves
        // mass. Off-grid velocities on a non-wrapping axis are 0 (no flux
        // through the wall) in the zero-flux forms; the BC forms take a
        // Dirichlet callable per quantity (field, and each velocity
        // component), exactly as Java does.

        // Value from a per-cell array at a possibly off-grid x: in-bounds
        // value, wrapped value if the axis wraps, otherwise 0 (zero-flux).
        // The 1D analogue of DisplacedX2D/DisplacedY2D.
        double Displaced1D(const double *pVals,
                           int32_t       x,
                           int32_t       xDim,
                           bool          wrapX);

        void Advection1(const double *pField,
                        double       *pDeltas,
                        const double *pXVels,
                        int32_t       xDim,
                        bool          wrapX);

        void Advection2(const double *pField,
                        double       *pDeltas,
                        const double *pXVels,
                        const double *pYVels,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        bool          wrapY);

        void Advection3(const double *pField,
                        double       *pDeltas,
                        const double *pXVels,
                        const double *pYVels,
                        const double *pZVels,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapX,
                        bool          wrapY,
                        bool          wrapZ);

        // Dirichlet-boundary lookups for the BC advection forms below:
        // like the zero-flux Displaced* functions, but an off-grid
        // coordinate on a non-wrapping axis evaluates the boundary
        // callable instead of returning 0.
        template <typename BCFn>
        double Displaced1D(const double *pVals,
                           int32_t       x,
                           int32_t       xDim,
                           bool          wrapX,
                           BCFn          bc)
        {
            if (InDim(x, xDim))
            {
                return pVals[x];
            }
            if (wrapX)
            {
                return pVals[Wrap(x, xDim)];
            }
            return bc(x);
        }

        template <typename BCFn>
        double DisplacedX2D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       xDim,
                            int32_t       yDim,
                            bool          wrapX,
                            BCFn          bc)
        {
            if (InDim(x, xDim))
            {
                return pVals[x * yDim + y];
            }
            if (wrapX)
            {
                return pVals[Wrap(x, xDim) * yDim + y];
            }
            return bc(x, y);
        }

        template <typename BCFn>
        double DisplacedY2D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       xDim,
                            int32_t       yDim,
                            bool          wrapY,
                            BCFn          bc)
        {
            if (InDim(y, yDim))
            {
                return pVals[x * yDim + y];
            }
            if (wrapY)
            {
                return pVals[x * yDim + Wrap(y, yDim)];
            }
            return bc(x, y);
        }

        template <typename BCFn>
        double DisplacedX3D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       z,
                            int32_t       xDim,
                            int32_t       yDim,
                            int32_t       zDim,
                            bool          wrapX,
                            BCFn          bc)
        {
            if (InDim(x, xDim))
            {
                return pVals[x * yDim * zDim + y * zDim + z];
            }
            if (wrapX)
            {
                return pVals[Wrap(x, xDim) * yDim * zDim + y * zDim + z];
            }
            return bc(x, y, z);
        }

        template <typename BCFn>
        double DisplacedY3D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       z,
                            int32_t       xDim,
                            int32_t       yDim,
                            int32_t       zDim,
                            bool          wrapY,
                            BCFn          bc)
        {
            if (InDim(y, yDim))
            {
                return pVals[x * yDim * zDim + y * zDim + z];
            }
            if (wrapY)
            {
                return pVals[x * yDim * zDim + Wrap(y, yDim) * zDim + z];
            }
            return bc(x, y, z);
        }

        template <typename BCFn>
        double DisplacedZ3D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       z,
                            int32_t       xDim,
                            int32_t       yDim,
                            int32_t       zDim,
                            bool          wrapZ,
                            BCFn          bc)
        {
            if (InDim(z, zDim))
            {
                return pVals[x * yDim * zDim + y * zDim + z];
            }
            if (wrapZ)
            {
                return pVals[x * yDim * zDim + y * zDim + Wrap(z, zDim)];
            }
            return bc(x, y, z);
        }

        template <typename BCFn, typename XVelBCFn>
        void Advection1(const double *pField,
                        double       *pDeltas,
                        const double *pXVels,
                        int32_t       xDim,
                        bool          wrapX,
                        BCFn          bc,
                        XVelBCFn      xVelBC)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                double xVelP = ((x == (xDim - 1)) && !wrapX) ? Displaced1D(pXVels, x + 1, xDim, wrapX, xVelBC) : pXVels[x];
                double xVelM = Displaced1D(pXVels, x - 1, xDim, wrapX, xVelBC);
                double centerVal = pField[x];
                double xFluxP = 0.0;
                double xFluxM = 0.0;
                if (xVelP > 0)
                {
                    xFluxP = xVelP * centerVal;
                }
                else if (xVelP < 0)
                {
                    xFluxP = xVelP * Displaced1D(pField, x + 1, xDim, wrapX, bc);
                }
                if (xVelM > 0)
                {
                    xFluxM = xVelM * Displaced1D(pField, x - 1, xDim, wrapX, bc);
                }
                else if (xVelM < 0)
                {
                    xFluxM = xVelM * centerVal;
                }
                pDeltas[x] += -xFluxP + xFluxM;
            }
        }

        template <typename BCFn, typename XVelBCFn, typename YVelBCFn>
        void Advection2(const double *pField,
                        double       *pDeltas,
                        const double *pXVels,
                        const double *pYVels,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        bool          wrapY,
                        BCFn          bc,
                        XVelBCFn      xVelBC,
                        YVelBCFn      yVelBC)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    int32_t i = x * yDim + y;
                    double  xVelP = ((x == (xDim - 1)) && !wrapX) ? DisplacedX2D(pXVels, x + 1, y, xDim, yDim, wrapX, xVelBC) : pXVels[i];
                    double  yVelP = ((y == (yDim - 1)) && !wrapY) ? DisplacedY2D(pYVels, x, y + 1, xDim, yDim, wrapY, yVelBC) : pYVels[i];
                    double  xVelM = DisplacedX2D(pXVels, x - 1, y, xDim, yDim, wrapX, xVelBC);
                    double  yVelM = DisplacedY2D(pYVels, x, y - 1, xDim, yDim, wrapY, yVelBC);
                    double  centerVal = pField[i];
                    double  xFluxP = 0.0;
                    double  xFluxM = 0.0;
                    double  yFluxP = 0.0;
                    double  yFluxM = 0.0;
                    if (xVelP > 0)
                    {
                        xFluxP = xVelP * centerVal;
                    }
                    else if (xVelP < 0)
                    {
                        xFluxP = xVelP * DisplacedX2D(pField, x + 1, y, xDim, yDim, wrapX, bc);
                    }
                    if (xVelM > 0)
                    {
                        xFluxM = xVelM * DisplacedX2D(pField, x - 1, y, xDim, yDim, wrapX, bc);
                    }
                    else if (xVelM < 0)
                    {
                        xFluxM = xVelM * centerVal;
                    }
                    if (yVelP > 0)
                    {
                        yFluxP = yVelP * centerVal;
                    }
                    else if (yVelP < 0)
                    {
                        yFluxP = yVelP * DisplacedY2D(pField, x, y + 1, xDim, yDim, wrapY, bc);
                    }
                    if (yVelM > 0)
                    {
                        yFluxM = yVelM * DisplacedY2D(pField, x, y - 1, xDim, yDim, wrapY, bc);
                    }
                    else if (yVelM < 0)
                    {
                        yFluxM = yVelM * centerVal;
                    }
                    pDeltas[i] += -xFluxP + xFluxM - yFluxP + yFluxM;
                }
            }
        }

        // z goes through DisplacedZ3D with the correct coordinates and
        // wrap flag — the Java original's z-handling bugs (fixed in the
        // zero-flux version in PDEequations.cpp) are fixed here too; see
        // docs/JAVA_BUGS_FOUND.md #18.
        template <typename BCFn, typename XVelBCFn, typename YVelBCFn, typename ZVelBCFn>
        void Advection3(const double *pField,
                        double       *pDeltas,
                        const double *pXVels,
                        const double *pYVels,
                        const double *pZVels,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapX,
                        bool          wrapY,
                        bool          wrapZ,
                        BCFn          bc,
                        XVelBCFn      xVelBC,
                        YVelBCFn      yVelBC,
                        ZVelBCFn      zVelBC)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    for (int32_t z = 0; z < zDim; z++)
                    {
                        int32_t i = x * yDim * zDim + y * zDim + z;
                        double  xVelP = ((x == (xDim - 1)) && !wrapX) ? DisplacedX3D(pXVels, x + 1, y, z, xDim, yDim, zDim, wrapX, xVelBC) : pXVels[i];
                        double  yVelP = ((y == (yDim - 1)) && !wrapY) ? DisplacedY3D(pYVels, x, y + 1, z, xDim, yDim, zDim, wrapY, yVelBC) : pYVels[i];
                        double  zVelP = ((z == (zDim - 1)) && !wrapZ) ? DisplacedZ3D(pZVels, x, y, z + 1, xDim, yDim, zDim, wrapZ, zVelBC) : pZVels[i];
                        double  xVelM = DisplacedX3D(pXVels, x - 1, y, z, xDim, yDim, zDim, wrapX, xVelBC);
                        double  yVelM = DisplacedY3D(pYVels, x, y - 1, z, xDim, yDim, zDim, wrapY, yVelBC);
                        double  zVelM = DisplacedZ3D(pZVels, x, y, z - 1, xDim, yDim, zDim, wrapZ, zVelBC);
                        double  centerVal = pField[i];
                        double  xFluxP = 0.0;
                        double  xFluxM = 0.0;
                        double  yFluxP = 0.0;
                        double  yFluxM = 0.0;
                        double  zFluxP = 0.0;
                        double  zFluxM = 0.0;
                        if (xVelP > 0)
                        {
                            xFluxP = xVelP * centerVal;
                        }
                        else if (xVelP < 0)
                        {
                            xFluxP = xVelP * DisplacedX3D(pField, x + 1, y, z, xDim, yDim, zDim, wrapX, bc);
                        }
                        if (xVelM > 0)
                        {
                            xFluxM = xVelM * DisplacedX3D(pField, x - 1, y, z, xDim, yDim, zDim, wrapX, bc);
                        }
                        else if (xVelM < 0)
                        {
                            xFluxM = xVelM * centerVal;
                        }
                        if (yVelP > 0)
                        {
                            yFluxP = yVelP * centerVal;
                        }
                        else if (yVelP < 0)
                        {
                            yFluxP = yVelP * DisplacedY3D(pField, x, y + 1, z, xDim, yDim, zDim, wrapY, bc);
                        }
                        if (yVelM > 0)
                        {
                            yFluxM = yVelM * DisplacedY3D(pField, x, y - 1, z, xDim, yDim, zDim, wrapY, bc);
                        }
                        else if (yVelM < 0)
                        {
                            yFluxM = yVelM * centerVal;
                        }
                        if (zVelP > 0)
                        {
                            zFluxP = zVelP * centerVal;
                        }
                        else if (zVelP < 0)
                        {
                            zFluxP = zVelP * DisplacedZ3D(pField, x, y, z + 1, xDim, yDim, zDim, wrapZ, bc);
                        }
                        if (zVelM > 0)
                        {
                            zFluxM = zVelM * DisplacedZ3D(pField, x, y, z - 1, xDim, yDim, zDim, wrapZ, bc);
                        }
                        else if (zVelM < 0)
                        {
                            zFluxM = zVelM * centerVal;
                        }
                        pDeltas[i] += -xFluxP + xFluxM - yFluxP + yFluxM - zFluxP + zFluxM;
                    }
                }
            }
        }
    }
}
