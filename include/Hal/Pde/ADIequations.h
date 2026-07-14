#pragma once

#include "Hal/MathUtil.h"
#include "Hal/Pde/TdmaSolver.h"

#include <cstdint>

namespace Hal
{
    namespace Internal
    {
        // Alternating-Direction-Implicit (Crank-Nicolson-style) implicit
        // diffusion solver, ported from Tools/Internal/ADIequations.java.
        // Unconditionally stable (no CFL cap), at the cost of a
        // tridiagonal solve per row/column/pencil each step. Each
        // dimension does one explicit half-step along the *other* axis
        // (axes, in 3D) followed by an implicit solve along the axis
        // currently being swept; 2D/3D sweep each axis in turn.
        //
        // For a periodic sinusoidal mode, one full sweep has an exact
        // closed-form eigenvalue — see PDEADITests.cpp — which is what
        // this port is verified against, the implicit-solver analogue of
        // the explicit stencil's eigenmode-decay check in PDEequations.h.

        // --- Delta helpers: Dirichlet boundary doubles the reflected
        // term (matching the Crank-Nicolson half-step's factor of 2),
        // rather than the explicit stencil's undoubled Dirichlet term. ---

        double Delta1DADI(const double *pField,
                          double        centerVal,
                          int32_t       x,
                          int32_t       xDim,
                          bool          wrapX);

        template <typename BCFn>
        double Delta1DADI(const double *pField,
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
            double bcVal = bc(x);
            return (bcVal - centerVal) * 2 + 2 * bcVal;
        }

        double DeltaX2DADI(const double *pField,
                           double        centerVal,
                           int32_t       x,
                           int32_t       y,
                           int32_t       xDim,
                           int32_t       yDim,
                           bool          wrapX);

        template <typename BCFn>
        double DeltaX2DADI(const double *pField,
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
            double bcVal = bc(x, y);
            return (bcVal - centerVal) * 2 + 2 * bcVal;
        }

        double DeltaY2DADI(const double *pField,
                           double        centerVal,
                           int32_t       x,
                           int32_t       y,
                           int32_t       xDim,
                           int32_t       yDim,
                           bool          wrapY);

        template <typename BCFn>
        double DeltaY2DADI(const double *pField,
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
            double bcVal = bc(x, y);
            return (bcVal - centerVal) * 2 + 2 * bcVal;
        }

        double DeltaX3DADI(const double *pField,
                           double        centerVal,
                           int32_t       x,
                           int32_t       y,
                           int32_t       z,
                           int32_t       xDim,
                           int32_t       yDim,
                           int32_t       zDim,
                           bool          wrapX);

        template <typename BCFn>
        double DeltaX3DADI(const double *pField,
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
            double bcVal = bc(x, y, z);
            return (bcVal - centerVal) * 2 + 2 * bcVal;
        }

        double DeltaY3DADI(const double *pField,
                           double        centerVal,
                           int32_t       x,
                           int32_t       y,
                           int32_t       z,
                           int32_t       xDim,
                           int32_t       yDim,
                           int32_t       zDim,
                           bool          wrapY);

        template <typename BCFn>
        double DeltaY3DADI(const double *pField,
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
            double bcVal = bc(x, y, z);
            return (bcVal - centerVal) * 2 + 2 * bcVal;
        }

        double DeltaZ3DADI(const double *pField,
                           double        centerVal,
                           int32_t       x,
                           int32_t       y,
                           int32_t       z,
                           int32_t       xDim,
                           int32_t       yDim,
                           int32_t       zDim,
                           bool          wrapZ);

        template <typename BCFn>
        double DeltaZ3DADI(const double *pField,
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
            double bcVal = bc(x, y, z);
            return (bcVal - centerVal) * 2 + 2 * bcVal;
        }

        // --- Explicit half-steps (used as TDMA's right-hand side) ---

        double ExplicitDiffusion1ADI(int32_t       x,
                                     const double *pField,
                                     double        diffRate,
                                     int32_t       xDim,
                                     bool          wrapX);

        template <typename BCFn>
        double ExplicitDiffusion1ADI(int32_t       x,
                                     const double *pField,
                                     double        diffRate,
                                     int32_t       xDim,
                                     bool          wrapX,
                                     BCFn          bc)
        {
            double centerVal = pField[x];
            double valSum = Delta1DADI(pField, centerVal, x + 1, xDim, wrapX, bc);
            valSum += Delta1DADI(pField, centerVal, x - 1, xDim, wrapX, bc);
            return centerVal + (valSum * diffRate) / 2;
        }

        double ExplicitDiffusionX2ADI(int32_t       x,
                                      int32_t       y,
                                      const double *pField,
                                      double        diffRate,
                                      int32_t       xDim,
                                      int32_t       yDim,
                                      bool          wrapX);

        template <typename BCFn>
        double ExplicitDiffusionX2ADI(int32_t       x,
                                      int32_t       y,
                                      const double *pField,
                                      double        diffRate,
                                      int32_t       xDim,
                                      int32_t       yDim,
                                      bool          wrapX,
                                      BCFn          bc)
        {
            int32_t i = x * yDim + y;
            double  centerVal = pField[i];
            double  deltaSum  = DeltaX2DADI(pField, centerVal, x + 1, y, xDim, yDim, wrapX, bc);
            deltaSum += DeltaX2DADI(pField, centerVal, x - 1, y, xDim, yDim, wrapX, bc);
            return centerVal + (deltaSum * diffRate) / 2;
        }

        double ExplicitDiffusionY2ADI(int32_t       x,
                                      int32_t       y,
                                      const double *pField,
                                      double        diffRate,
                                      int32_t       xDim,
                                      int32_t       yDim,
                                      bool          wrapY);

        template <typename BCFn>
        double ExplicitDiffusionY2ADI(int32_t       x,
                                      int32_t       y,
                                      const double *pField,
                                      double        diffRate,
                                      int32_t       xDim,
                                      int32_t       yDim,
                                      bool          wrapY,
                                      BCFn          bc)
        {
            int32_t i = x * yDim + y;
            double  centerVal = pField[i];
            double  deltaSum  = DeltaY2DADI(pField, centerVal, x, y + 1, xDim, yDim, wrapY, bc);
            deltaSum += DeltaY2DADI(pField, centerVal, x, y - 1, xDim, yDim, wrapY, bc);
            return centerVal + (deltaSum * diffRate) / 2;
        }

        double ExplicitDiffusionYZ3ADI(int32_t       x,
                                       int32_t       y,
                                       int32_t       z,
                                       const double *pField,
                                       double        diffRate,
                                       int32_t       xDim,
                                       int32_t       yDim,
                                       int32_t       zDim,
                                       bool          wrapY,
                                       bool          wrapZ);

        template <typename BCFn>
        double ExplicitDiffusionYZ3ADI(int32_t       x,
                                       int32_t       y,
                                       int32_t       z,
                                       const double *pField,
                                       double        diffRate,
                                       int32_t       xDim,
                                       int32_t       yDim,
                                       int32_t       zDim,
                                       bool          wrapY,
                                       bool          wrapZ,
                                       BCFn          bc)
        {
            int32_t i = x * yDim * zDim + y * zDim + z;
            double  centerVal = pField[i];
            double  deltaSum  = DeltaY3DADI(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY, bc);
            deltaSum += DeltaY3DADI(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY, bc);
            deltaSum += DeltaZ3DADI(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ, bc);
            deltaSum += DeltaZ3DADI(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ, bc);
            return centerVal + (deltaSum * diffRate) / 3;
        }

        double ExplicitDiffusionXZ3ADI(int32_t       x,
                                       int32_t       y,
                                       int32_t       z,
                                       const double *pField,
                                       double        diffRate,
                                       int32_t       xDim,
                                       int32_t       yDim,
                                       int32_t       zDim,
                                       bool          wrapX,
                                       bool          wrapZ);

        template <typename BCFn>
        double ExplicitDiffusionXZ3ADI(int32_t       x,
                                       int32_t       y,
                                       int32_t       z,
                                       const double *pField,
                                       double        diffRate,
                                       int32_t       xDim,
                                       int32_t       yDim,
                                       int32_t       zDim,
                                       bool          wrapX,
                                       bool          wrapZ,
                                       BCFn          bc)
        {
            int32_t i = x * yDim * zDim + y * zDim + z;
            double  centerVal = pField[i];
            double  deltaSum  = DeltaX3DADI(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX, bc);
            deltaSum += DeltaX3DADI(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX, bc);
            deltaSum += DeltaZ3DADI(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ, bc);
            deltaSum += DeltaZ3DADI(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ, bc);
            return centerVal + (deltaSum * diffRate) / 3;
        }

        double ExplicitDiffusionXY3ADI(int32_t       x,
                                       int32_t       y,
                                       int32_t       z,
                                       const double *pField,
                                       double        diffRate,
                                       int32_t       xDim,
                                       int32_t       yDim,
                                       int32_t       zDim,
                                       bool          wrapX,
                                       bool          wrapY);

        template <typename BCFn>
        double ExplicitDiffusionXY3ADI(int32_t       x,
                                       int32_t       y,
                                       int32_t       z,
                                       const double *pField,
                                       double        diffRate,
                                       int32_t       xDim,
                                       int32_t       yDim,
                                       int32_t       zDim,
                                       bool          wrapX,
                                       bool          wrapY,
                                       BCFn          bc)
        {
            int32_t i = x * yDim * zDim + y * zDim + z;
            double  centerVal = pField[i];
            double  deltaSum  = DeltaX3DADI(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX, bc);
            deltaSum += DeltaX3DADI(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX, bc);
            deltaSum += DeltaY3DADI(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY, bc);
            deltaSum += DeltaY3DADI(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY, bc);
            return centerVal + (deltaSum * diffRate) / 3;
        }

        // Builds and solves the tridiagonal system for one row/column/
        // pencil: periodic (TDMAperiodic) if wrapDim, else zero-flux or
        // Dirichlet matrix coefficients depending on hasBC (a flag, not
        // the BC callable itself — the callable already fed into getIn's
        // explicit half-step; the matrix coefficients only need to know
        // whether *some* boundary condition is active).
        template <typename GetInFn, typename SetOutFn>
        void ADISolveRow(int32_t     lenToSolve,
                         double      diffRate,
                         double      diffScaleFactor,
                         bool        wrapDim,
                         bool        hasBC,
                         TdmaSolver &tdma,
                         GetInFn     getIn,
                         SetOutFn    setOut)
        {
            if (wrapDim)
            {
                tdma.TDMAperiodic(
                    lenToSolve, getIn, setOut,
                    [&](int32_t)
                    { return 1 + diffRate * 2 / diffScaleFactor; },
                    [&](int32_t)
                    { return -diffRate / diffScaleFactor; },
                    [&](int32_t)
                    { return -diffRate / diffScaleFactor; });
                return;
            }
            if (!hasBC)
            {
                tdma.TDMA(
                    lenToSolve, getIn, setOut,
                    [&](int32_t i)
                    {
                        if ((i == 0) || (i == (lenToSolve - 1)))
                        {
                            return 1 + diffRate / diffScaleFactor;
                        }
                        return 1 + diffRate * 2 / diffScaleFactor;
                    },
                    [&](int32_t)
                    { return -diffRate / diffScaleFactor; },
                    [&](int32_t)
                    { return -diffRate / diffScaleFactor; });
            }
            else
            {
                tdma.TDMA(
                    lenToSolve, getIn, setOut,
                    [&](int32_t i)
                    {
                        if ((i == 0) || (i == (lenToSolve - 1)))
                        {
                            return 1 + diffRate * 3 / diffScaleFactor;
                        }
                        return 1 + diffRate * 2 / diffScaleFactor;
                    },
                    [&](int32_t)
                    { return -diffRate / diffScaleFactor; },
                    [&](int32_t)
                    { return -diffRate / diffScaleFactor; });
            }
        }

        // --- 1D Crank-Nicolson / 2D-3D ADI diffusion ---

        inline void Diffusion1ADI(const double *pField,
                                  double       *pDeltas,
                                  double        diffRate,
                                  int32_t       xDim,
                                  bool          wrapX,
                                  TdmaSolver   &tdma)
        {
            ADISolveRow(
                xDim, diffRate, 2, wrapX, false, tdma,
                [&](int32_t i)
                { return ExplicitDiffusion1ADI(i, pField, diffRate, xDim, wrapX); },
                [&](int32_t i, double v)
                { pDeltas[i] += v - pField[i]; });
        }

        template <typename BCFn>
        void Diffusion1ADI(const double *pField,
                           double       *pDeltas,
                           double        diffRate,
                           int32_t       xDim,
                           bool          wrapX,
                           BCFn          bc,
                           TdmaSolver   &tdma)
        {
            ADISolveRow(
                xDim, diffRate, 2, wrapX, true, tdma,
                [&](int32_t i)
                { return ExplicitDiffusion1ADI(i, pField, diffRate, xDim, wrapX, bc); },
                [&](int32_t i, double v)
                { pDeltas[i] += v - pField[i]; });
        }

        inline void Diffusion2DADI(const double *pField,
                                   double       *pScratch,
                                   double       *pDeltas,
                                   double        diffRate,
                                   int32_t       xDim,
                                   int32_t       yDim,
                                   bool          wrapX,
                                   bool          wrapY,
                                   TdmaSolver   &tdma)
        {
            for (int32_t y = 0; y < yDim; y++)
            {
                ADISolveRow(
                    xDim, diffRate, 2, wrapX, false, tdma,
                    [&, y](int32_t i)
                    { return ExplicitDiffusionY2ADI(i, y, pField, diffRate, xDim, yDim, wrapY); },
                    [&, y](int32_t i, double v)
                    { pScratch[i * yDim + y] = v; });
            }
            for (int32_t x = 0; x < xDim; x++)
            {
                ADISolveRow(
                    yDim, diffRate, 2, wrapY, false, tdma,
                    [&, x](int32_t i)
                    { return ExplicitDiffusionX2ADI(x, i, pScratch, diffRate, xDim, yDim, wrapX); },
                    [&, x](int32_t i, double v)
                    {
                        int32_t index = x * yDim + i;
                        pDeltas[index] += v - pField[index];
                    });
            }
        }

        template <typename BCFn>
        void Diffusion2DADI(const double *pField,
                            double       *pScratch,
                            double       *pDeltas,
                            double        diffRate,
                            int32_t       xDim,
                            int32_t       yDim,
                            bool          wrapX,
                            bool          wrapY,
                            BCFn          bc,
                            TdmaSolver   &tdma)
        {
            for (int32_t y = 0; y < yDim; y++)
            {
                ADISolveRow(
                    xDim, diffRate, 2, wrapX, true, tdma,
                    [&, y](int32_t i)
                    { return ExplicitDiffusionY2ADI(i, y, pField, diffRate, xDim, yDim, wrapY, bc); },
                    [&, y](int32_t i, double v)
                    { pScratch[i * yDim + y] = v; });
            }
            for (int32_t x = 0; x < xDim; x++)
            {
                ADISolveRow(
                    yDim, diffRate, 2, wrapY, true, tdma,
                    [&, x](int32_t i)
                    { return ExplicitDiffusionX2ADI(x, i, pScratch, diffRate, xDim, yDim, wrapX, bc); },
                    [&, x](int32_t i, double v)
                    {
                        int32_t index = x * yDim + i;
                        pDeltas[index] += v - pField[index];
                    });
            }
        }

        inline void Diffusion3DADI(const double *pField,
                                   double       *pScratch1,
                                   double       *pScratch2,
                                   double       *pDeltas,
                                   double        diffRate,
                                   int32_t       xDim,
                                   int32_t       yDim,
                                   int32_t       zDim,
                                   bool          wrapX,
                                   bool          wrapY,
                                   bool          wrapZ,
                                   TdmaSolver   &tdma)
        {
            for (int32_t y = 0; y < yDim; y++)
            {
                for (int32_t z = 0; z < zDim; z++)
                {
                    ADISolveRow(
                        xDim, diffRate, 3, wrapX, false, tdma,
                        [&, y, z](int32_t i)
                        { return ExplicitDiffusionYZ3ADI(i, y, z, pField, diffRate, xDim, yDim, zDim, wrapY, wrapZ); },
                        [&, y, z](int32_t i, double v)
                        { pScratch1[i * yDim * zDim + y * zDim + z] = v; });
                }
            }
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t z = 0; z < zDim; z++)
                {
                    ADISolveRow(
                        yDim, diffRate, 3, wrapY, false, tdma,
                        [&, x, z](int32_t i)
                        { return ExplicitDiffusionXZ3ADI(x, i, z, pScratch1, diffRate, xDim, yDim, zDim, wrapX, wrapZ); },
                        [&, x, z](int32_t i, double v)
                        { pScratch2[x * yDim * zDim + i * zDim + z] = v; });
                }
            }
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    ADISolveRow(
                        zDim, diffRate, 3, wrapZ, false, tdma,
                        [&, x, y](int32_t i)
                        { return ExplicitDiffusionXY3ADI(x, y, i, pScratch2, diffRate, xDim, yDim, zDim, wrapX, wrapY); },
                        [&, x, y](int32_t i, double v)
                        {
                            int32_t index = x * yDim * zDim + y * zDim + i;
                            pDeltas[index] += v - pField[index];
                        });
                }
            }
        }

        template <typename BCFn>
        void Diffusion3DADI(const double *pField,
                            double       *pScratch1,
                            double       *pScratch2,
                            double       *pDeltas,
                            double        diffRate,
                            int32_t       xDim,
                            int32_t       yDim,
                            int32_t       zDim,
                            bool          wrapX,
                            bool          wrapY,
                            bool          wrapZ,
                            BCFn          bc,
                            TdmaSolver   &tdma)
        {
            for (int32_t y = 0; y < yDim; y++)
            {
                for (int32_t z = 0; z < zDim; z++)
                {
                    ADISolveRow(
                        xDim, diffRate, 3, wrapX, true, tdma,
                        [&, y, z](int32_t i)
                        { return ExplicitDiffusionYZ3ADI(i, y, z, pField, diffRate, xDim, yDim, zDim, wrapY, wrapZ, bc); },
                        [&, y, z](int32_t i, double v)
                        { pScratch1[i * yDim * zDim + y * zDim + z] = v; });
                }
            }
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t z = 0; z < zDim; z++)
                {
                    ADISolveRow(
                        yDim, diffRate, 3, wrapY, true, tdma,
                        [&, x, z](int32_t i)
                        { return ExplicitDiffusionXZ3ADI(x, i, z, pScratch1, diffRate, xDim, yDim, zDim, wrapX, wrapZ, bc); },
                        [&, x, z](int32_t i, double v)
                        { pScratch2[x * yDim * zDim + i * zDim + z] = v; });
                }
            }
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    ADISolveRow(
                        zDim, diffRate, 3, wrapZ, true, tdma,
                        [&, x, y](int32_t i)
                        { return ExplicitDiffusionXY3ADI(x, y, i, pScratch2, diffRate, xDim, yDim, zDim, wrapX, wrapY, bc); },
                        [&, x, y](int32_t i, double v)
                        {
                            int32_t index = x * yDim * zDim + y * zDim + i;
                            pDeltas[index] += v - pField[index];
                        });
                }
            }
        }
    }
}
