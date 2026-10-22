#pragma once

#include "Hal/MathUtil.h"
#include "Hal/Pde/PDEequations.h"
#include "Hal/Pde/TdmaSolver.h"

#include <cstdint>

namespace Hal
{
    namespace Internal
    {
        // Alternating-direction-implicit diffusion. Each sweep solves one
        // axis implicitly (a tridiagonal system per line) with an explicit
        // step along the other axes as the right-hand side; each half
        // applies diffRate / diffScaleFactor (2 in 1D/2D, 3 in 3D). 1D is
        // Crank-Nicolson. Unconditionally stable.
        //
        // Dirichlet boundaries use a ghost cell reflecting the boundary
        // value: the explicit terms use the stencil's (bc - center) * 2, and
        // the implicitly solved axis adds its known boundary term to the
        // right-hand side (ImplicitBoundarySource).

        // --- Explicit half-steps (the right-hand side of each sweep) ---

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
            double valSum = Delta1D(pField, centerVal, x + 1, xDim, wrapX, bc);
            valSum += Delta1D(pField, centerVal, x - 1, xDim, wrapX, bc);
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
            double  deltaSum  = DeltaX2D(pField, centerVal, x + 1, y, xDim, yDim, wrapX, bc);
            deltaSum += DeltaX2D(pField, centerVal, x - 1, y, xDim, yDim, wrapX, bc);
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
            double  deltaSum  = DeltaY2D(pField, centerVal, x, y + 1, xDim, yDim, wrapY, bc);
            deltaSum += DeltaY2D(pField, centerVal, x, y - 1, xDim, yDim, wrapY, bc);
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
            double  deltaSum  = DeltaY3D(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY, bc);
            deltaSum += DeltaY3D(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY, bc);
            deltaSum += DeltaZ3D(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ, bc);
            deltaSum += DeltaZ3D(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ, bc);
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
            double  deltaSum  = DeltaX3D(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX, bc);
            deltaSum += DeltaX3D(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX, bc);
            deltaSum += DeltaZ3D(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ, bc);
            deltaSum += DeltaZ3D(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ, bc);
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
            double  deltaSum  = DeltaX3D(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX, bc);
            deltaSum += DeltaX3D(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX, bc);
            deltaSum += DeltaY3D(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY, bc);
            deltaSum += DeltaY3D(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY, bc);
            return centerVal + (deltaSum * diffRate) / 3;
        }

        // Right-hand-side term for a Dirichlet boundary on the implicitly
        // solved axis. i indexes a line of length len; edgeBc(-1) and
        // edgeBc(len) give the boundary values beyond each end.
        template <typename EdgeBCFn>
        double ImplicitBoundarySource(int32_t  i,
                                      int32_t  len,
                                      bool     wrap,
                                      double   diffRate,
                                      double   diffScaleFactor,
                                      EdgeBCFn edgeBc)
        {
            if (wrap)
            {
                return 0.0;
            }
            double source = 0.0;
            if (i == 0)
            {
                source += edgeBc(-1);
            }
            if (i == (len - 1))
            {
                source += edgeBc(len);
            }
            return source * 2.0 * diffRate / diffScaleFactor;
        }

        // Builds and solves the tridiagonal system for one line: periodic if
        // wrapDim, otherwise with zero-flux or (hasBC) Dirichlet ghost-cell
        // coefficients at the ends. getIn supplies the right-hand side.
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
            double offDiagonal = -diffRate / diffScaleFactor;
            if (wrapDim)
            {
                tdma.TDMAperiodic(
                    lenToSolve, getIn, setOut,
                    [&](int32_t)
                    { return 1 - 2 * offDiagonal; },
                    [&](int32_t)
                    { return offDiagonal; },
                    [&](int32_t)
                    { return offDiagonal; });
                return;
            }
            // Each in-line neighbor weighs 1; past an end, a Dirichlet ghost
            // weighs 2 and a zero-flux edge 0.
            double endAdjust = hasBC ? 1.0 : -1.0;
            tdma.TDMA(
                lenToSolve, getIn, setOut,
                [&](int32_t i)
                {
                    double weight = 2.0;
                    if (i == 0)
                    {
                        weight += endAdjust;
                    }
                    if (i == (lenToSolve - 1))
                    {
                        weight += endAdjust;
                    }
                    return 1 - weight * offDiagonal;
                },
                [&](int32_t)
                { return offDiagonal; },
                [&](int32_t)
                { return offDiagonal; });
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
                { return ExplicitDiffusion1ADI(i, pField, diffRate, xDim, wrapX, bc) + ImplicitBoundarySource(i, xDim, wrapX, diffRate, 2, bc); },
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
                    {
                        return ExplicitDiffusionY2ADI(i, y, pField, diffRate, xDim, yDim, wrapY, bc) +
                               ImplicitBoundarySource(i, xDim, wrapX, diffRate, 2, [&, y](int32_t xEdge) { return bc(xEdge, y); });
                    },
                    [&, y](int32_t i, double v)
                    { pScratch[i * yDim + y] = v; });
            }
            for (int32_t x = 0; x < xDim; x++)
            {
                ADISolveRow(
                    yDim, diffRate, 2, wrapY, true, tdma,
                    [&, x](int32_t i)
                    {
                        return ExplicitDiffusionX2ADI(x, i, pScratch, diffRate, xDim, yDim, wrapX, bc) +
                               ImplicitBoundarySource(i, yDim, wrapY, diffRate, 2, [&, x](int32_t yEdge) { return bc(x, yEdge); });
                    },
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
                        {
                            return ExplicitDiffusionYZ3ADI(i, y, z, pField, diffRate, xDim, yDim, zDim, wrapY, wrapZ, bc) +
                                   ImplicitBoundarySource(i, xDim, wrapX, diffRate, 3, [&, y, z](int32_t xEdge) { return bc(xEdge, y, z); });
                        },
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
                        {
                            return ExplicitDiffusionXZ3ADI(x, i, z, pScratch1, diffRate, xDim, yDim, zDim, wrapX, wrapZ, bc) +
                                   ImplicitBoundarySource(i, yDim, wrapY, diffRate, 3, [&, x, z](int32_t yEdge) { return bc(x, yEdge, z); });
                        },
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
                        {
                            return ExplicitDiffusionXY3ADI(x, y, i, pScratch2, diffRate, xDim, yDim, zDim, wrapX, wrapY, bc) +
                                   ImplicitBoundarySource(i, zDim, wrapZ, diffRate, 3, [&, x, y](int32_t zEdge) { return bc(x, y, zEdge); });
                        },
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
