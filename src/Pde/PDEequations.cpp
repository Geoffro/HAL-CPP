#include "Hal/Pde/PDEequations.h"

namespace Hal
{
    namespace Internal
    {
        double Delta1D(const double *pField,
                       double        centerVal,
                       int32_t       x,
                       int32_t       xDim,
                       bool          wrapX)
        {
            if (InDim(x, xDim))
            {
                return pField[x] - centerVal;
            }
            if (wrapX)
            {
                return pField[Wrap(x, xDim)] - centerVal;
            }
            return 0;
        }

        double DeltaX2D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX)
        {
            if (InDim(x, xDim))
            {
                return pField[x * yDim + y] - centerVal;
            }
            if (wrapX)
            {
                return pField[Wrap(x, xDim) * yDim + y] - centerVal;
            }
            return 0;
        }

        double DeltaY2D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapY)
        {
            if (InDim(y, yDim))
            {
                return pField[x * yDim + y] - centerVal;
            }
            if (wrapY)
            {
                return pField[x * yDim + Wrap(y, yDim)] - centerVal;
            }
            return 0;
        }

        double DeltaX3D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       z,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapX)
        {
            if (InDim(x, xDim))
            {
                return pField[x * yDim * zDim + y * zDim + z] - centerVal;
            }
            if (wrapX)
            {
                return pField[Wrap(x, xDim) * yDim * zDim + y * zDim + z] - centerVal;
            }
            return 0;
        }

        double DeltaY3D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       z,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapY)
        {
            if (InDim(y, yDim))
            {
                return pField[x * yDim * zDim + y * zDim + z] - centerVal;
            }
            if (wrapY)
            {
                return pField[x * yDim * zDim + Wrap(y, yDim) * zDim + z] - centerVal;
            }
            return 0;
        }

        double DeltaZ3D(const double *pField,
                        double        centerVal,
                        int32_t       x,
                        int32_t       y,
                        int32_t       z,
                        int32_t       xDim,
                        int32_t       yDim,
                        int32_t       zDim,
                        bool          wrapZ)
        {
            if (InDim(z, zDim))
            {
                return pField[x * yDim * zDim + y * zDim + z] - centerVal;
            }
            if (wrapZ)
            {
                return pField[x * yDim * zDim + y * zDim + Wrap(z, zDim)] - centerVal;
            }
            return 0;
        }

        void Diffusion1(const double *pField,
                        double       *pDeltas,
                        double        diffRate,
                        int32_t       xDim,
                        bool          wrapX)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                double centerVal = pField[x];
                double valSum = Delta1D(pField, centerVal, x + 1, xDim, wrapX);
                valSum += Delta1D(pField, centerVal, x - 1, xDim, wrapX);
                pDeltas[x] += valSum * diffRate;
            }
        }

        void Diffusion2(const double *pField,
                        double       *pDeltas,
                        double        diffRate,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        bool          wrapY)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    int32_t i = x * yDim + y;
                    double  centerVal = pField[i];
                    double  deltaSum  = DeltaX2D(pField, centerVal, x + 1, y, xDim, yDim, wrapX);
                    deltaSum += DeltaX2D(pField, centerVal, x - 1, y, xDim, yDim, wrapX);
                    deltaSum += DeltaY2D(pField, centerVal, x, y + 1, xDim, yDim, wrapY);
                    deltaSum += DeltaY2D(pField, centerVal, x, y - 1, xDim, yDim, wrapY);
                    pDeltas[i] += deltaSum * diffRate;
                }
            }
        }

        double DisplacedX2D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       xDim,
                            int32_t       yDim,
                            bool          wrapX)
        {
            if (InDim(x, xDim))
            {
                return pVals[x * yDim + y];
            }
            if (wrapX)
            {
                return pVals[Wrap(x, xDim) * yDim + y];
            }
            return 0.0;
        }

        double DisplacedY2D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       xDim,
                            int32_t       yDim,
                            bool          wrapY)
        {
            if (InDim(y, yDim))
            {
                return pVals[x * yDim + y];
            }
            if (wrapY)
            {
                return pVals[x * yDim + Wrap(y, yDim)];
            }
            return 0.0;
        }

        void Diffusion2(const double *pField,
                        double       *pDeltas,
                        const double *pDiffRatesX,
                        const double *pDiffRatesY,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        bool          wrapY)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    int32_t i = x * yDim + y;
                    double  centerVal = pField[i];
                    // Per-face rates, matching the Java stencil: the +x/+y
                    // face uses the center cell's rate except at a non-wrap
                    // boundary (where the off-grid rate, 0, applies); the
                    // -x/-y face uses the neighbor cell's rate.
                    double dp1x = ((x == (xDim - 1)) && !wrapX) ? DisplacedX2D(pDiffRatesX, x + 1, y, xDim, yDim, wrapX) : pDiffRatesX[i];
                    double dm1x = DisplacedX2D(pDiffRatesX, x - 1, y, xDim, yDim, wrapX);
                    double dp1y = ((y == (yDim - 1)) && !wrapY) ? DisplacedY2D(pDiffRatesY, x, y + 1, xDim, yDim, wrapY) : pDiffRatesY[i];
                    double dm1y = DisplacedY2D(pDiffRatesY, x, y - 1, xDim, yDim, wrapY);

                    double deltaSum = DeltaX2D(pField, centerVal, x + 1, y, xDim, yDim, wrapX) * dp1x;
                    deltaSum += DeltaX2D(pField, centerVal, x - 1, y, xDim, yDim, wrapX) * dm1x;
                    deltaSum += DeltaY2D(pField, centerVal, x, y + 1, xDim, yDim, wrapY) * dp1y;
                    deltaSum += DeltaY2D(pField, centerVal, x, y - 1, xDim, yDim, wrapY) * dm1y;
                    pDeltas[i] += deltaSum;
                }
            }
        }

        double DisplacedX3D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       z,
                            int32_t       xDim,
                            int32_t       yDim,
                            int32_t       zDim,
                            bool          wrapX)
        {
            if (InDim(x, xDim))
            {
                return pVals[x * yDim * zDim + y * zDim + z];
            }
            if (wrapX)
            {
                return pVals[Wrap(x, xDim) * yDim * zDim + y * zDim + z];
            }
            return 0.0;
        }

        double DisplacedY3D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       z,
                            int32_t       xDim,
                            int32_t       yDim,
                            int32_t       zDim,
                            bool          wrapY)
        {
            if (InDim(y, yDim))
            {
                return pVals[x * yDim * zDim + y * zDim + z];
            }
            if (wrapY)
            {
                return pVals[x * yDim * zDim + Wrap(y, yDim) * zDim + z];
            }
            return 0.0;
        }

        double DisplacedZ3D(const double *pVals,
                            int32_t       x,
                            int32_t       y,
                            int32_t       z,
                            int32_t       xDim,
                            int32_t       yDim,
                            int32_t       zDim,
                            bool          wrapZ)
        {
            if (InDim(z, zDim))
            {
                return pVals[x * yDim * zDim + y * zDim + z];
            }
            if (wrapZ)
            {
                return pVals[x * yDim * zDim + y * zDim + Wrap(z, zDim)];
            }
            return 0.0;
        }

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
                        bool          wrapZ)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    for (int32_t z = 0; z < zDim; z++)
                    {
                        int32_t i = x * yDim * zDim + y * zDim + z;
                        double  centerVal = pField[i];
                        // Per-face rates, matching the Java stencil: the +x/+y/+z
                        // face uses the center cell's rate except at a non-wrap
                        // boundary (off-grid rate 0); the -x/-y/-z face uses the
                        // neighbor cell's rate.
                        double dp1x = ((x == (xDim - 1)) && !wrapX) ? DisplacedX3D(pDiffRatesX, x + 1, y, z, xDim, yDim, zDim, wrapX) : pDiffRatesX[i];
                        double dm1x = DisplacedX3D(pDiffRatesX, x - 1, y, z, xDim, yDim, zDim, wrapX);
                        double dp1y = ((y == (yDim - 1)) && !wrapY) ? DisplacedY3D(pDiffRatesY, x, y + 1, z, xDim, yDim, zDim, wrapY) : pDiffRatesY[i];
                        double dm1y = DisplacedY3D(pDiffRatesY, x, y - 1, z, xDim, yDim, zDim, wrapY);
                        double dp1z = ((z == (zDim - 1)) && !wrapZ) ? DisplacedZ3D(pDiffRatesZ, x, y, z + 1, xDim, yDim, zDim, wrapZ) : pDiffRatesZ[i];
                        double dm1z = DisplacedZ3D(pDiffRatesZ, x, y, z - 1, xDim, yDim, zDim, wrapZ);

                        double deltaSum = DeltaX3D(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX) * dp1x;
                        deltaSum += DeltaX3D(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX) * dm1x;
                        deltaSum += DeltaY3D(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY) * dp1y;
                        deltaSum += DeltaY3D(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY) * dm1y;
                        deltaSum += DeltaZ3D(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ) * dp1z;
                        deltaSum += DeltaZ3D(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ) * dm1z;
                        pDeltas[i] += deltaSum;
                    }
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
                        bool          wrapZ)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    for (int32_t z = 0; z < zDim; z++)
                    {
                        int32_t i = x * yDim * zDim + y * zDim + z;
                        double  centerVal = pField[i];
                        double  deltaSum  = DeltaX3D(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX);
                        deltaSum += DeltaX3D(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX);
                        deltaSum += DeltaY3D(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY);
                        deltaSum += DeltaY3D(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY);
                        deltaSum += DeltaZ3D(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ);
                        deltaSum += DeltaZ3D(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ);
                        pDeltas[i] += deltaSum * diffRate;
                    }
                }
            }
        }

        void Advection1(const double *pField,
                        double       *pDeltas,
                        double        vel,
                        int32_t       xDim,
                        bool          wrapX)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                if (vel > 0)
                {
                    pDeltas[x] += Delta1D(pField, pField[x], x - 1, xDim, wrapX) * vel;
                }
                else if (vel < 0)
                {
                    pDeltas[x] -= Delta1D(pField, pField[x], x + 1, xDim, wrapX) * vel;
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
                        bool          wrapY)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    int32_t i = x * yDim + y;
                    double  centerVal = pField[i];
                    if (xVel > 0)
                    {
                        pDeltas[i] += DeltaX2D(pField, centerVal, x - 1, y, xDim, yDim, wrapX) * xVel;
                    }
                    else if (xVel < 0)
                    {
                        pDeltas[i] -= DeltaX2D(pField, centerVal, x + 1, y, xDim, yDim, wrapX) * xVel;
                    }
                    if (yVel > 0)
                    {
                        pDeltas[i] += DeltaY2D(pField, centerVal, x, y - 1, xDim, yDim, wrapY) * yVel;
                    }
                    else if (yVel < 0)
                    {
                        pDeltas[i] -= DeltaY2D(pField, centerVal, x, y + 1, xDim, yDim, wrapY) * yVel;
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
                        bool          wrapZ)
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
                            pDeltas[i] += DeltaX3D(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX) * xVel;
                        }
                        else if (xVel < 0)
                        {
                            pDeltas[i] -= DeltaX3D(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX) * xVel;
                        }
                        if (yVel > 0)
                        {
                            pDeltas[i] += DeltaY3D(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY) * yVel;
                        }
                        else if (yVel < 0)
                        {
                            pDeltas[i] -= DeltaY3D(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY) * yVel;
                        }
                        if (zVel > 0)
                        {
                            pDeltas[i] += DeltaZ3D(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ) * zVel;
                        }
                        else if (zVel < 0)
                        {
                            pDeltas[i] -= DeltaZ3D(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ) * zVel;
                        }
                    }
                }
            }
        }

        double Displaced1D(const double *pVals,
                           int32_t       x,
                           int32_t       xDim,
                           bool          wrapX)
        {
            if (InDim(x, xDim))
            {
                return pVals[x];
            }
            if (wrapX)
            {
                return pVals[Wrap(x, xDim)];
            }
            return 0.0;
        }

        void Advection1(const double *pField,
                        double       *pDeltas,
                        const double *pXVels,
                        int32_t       xDim,
                        bool          wrapX)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                // Upwind flux through each cell face: the +x face uses the
                // center cell's velocity except at a non-wrap boundary
                // (where the off-grid velocity, 0, applies — no flux
                // through the wall); the -x face uses the neighbor's.
                double xVelP = ((x == (xDim - 1)) && !wrapX) ? Displaced1D(pXVels, x + 1, xDim, wrapX) : pXVels[x];
                double xVelM = Displaced1D(pXVels, x - 1, xDim, wrapX);
                double centerVal = pField[x];
                double xFluxP = 0.0;
                double xFluxM = 0.0;
                if (xVelP > 0)
                {
                    xFluxP = xVelP * centerVal;
                }
                else if (xVelP < 0)
                {
                    xFluxP = xVelP * Displaced1D(pField, x + 1, xDim, wrapX);
                }
                if (xVelM > 0)
                {
                    xFluxM = xVelM * Displaced1D(pField, x - 1, xDim, wrapX);
                }
                else if (xVelM < 0)
                {
                    xFluxM = xVelM * centerVal;
                }
                pDeltas[x] += -xFluxP + xFluxM;
            }
        }

        void Advection2(const double *pField,
                        double       *pDeltas,
                        const double *pXVels,
                        const double *pYVels,
                        int32_t       xDim,
                        int32_t       yDim,
                        bool          wrapX,
                        bool          wrapY)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    int32_t i = x * yDim + y;
                    double  xVelP = ((x == (xDim - 1)) && !wrapX) ? DisplacedX2D(pXVels, x + 1, y, xDim, yDim, wrapX) : pXVels[i];
                    double  yVelP = ((y == (yDim - 1)) && !wrapY) ? DisplacedY2D(pYVels, x, y + 1, xDim, yDim, wrapY) : pYVels[i];
                    double  xVelM = DisplacedX2D(pXVels, x - 1, y, xDim, yDim, wrapX);
                    double  yVelM = DisplacedY2D(pYVels, x, y - 1, xDim, yDim, wrapY);
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
                        xFluxP = xVelP * DisplacedX2D(pField, x + 1, y, xDim, yDim, wrapX);
                    }
                    if (xVelM > 0)
                    {
                        xFluxM = xVelM * DisplacedX2D(pField, x - 1, y, xDim, yDim, wrapX);
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
                        yFluxP = yVelP * DisplacedY2D(pField, x, y + 1, xDim, yDim, wrapY);
                    }
                    if (yVelM > 0)
                    {
                        yFluxM = yVelM * DisplacedY2D(pField, x, y - 1, xDim, yDim, wrapY);
                    }
                    else if (yVelM < 0)
                    {
                        yFluxM = yVelM * centerVal;
                    }
                    pDeltas[i] += -xFluxP + xFluxM - yFluxP + yFluxM;
                }
            }
        }

        // NOTE: the Java original's z-direction handling here was broken
        // three ways (docs/JAVA_BUGS_FOUND.md #18): it looked z-neighbors
        // up through the Y-axis helper (which bounds-checks y, so an
        // off-grid z silently mis-indexed into the next row — or clean off
        // the end of the array), read the z-minus face velocity from the
        // center cell instead of z-1, and passed wrapX where wrapZ
        // belongs. All fixed below: z goes through DisplacedZ3D with the
        // correct coordinates and wrap flag, symmetric with x and y.
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
                        bool          wrapZ)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                for (int32_t y = 0; y < yDim; y++)
                {
                    for (int32_t z = 0; z < zDim; z++)
                    {
                        int32_t i = x * yDim * zDim + y * zDim + z;
                        double  xVelP = ((x == (xDim - 1)) && !wrapX) ? DisplacedX3D(pXVels, x + 1, y, z, xDim, yDim, zDim, wrapX) : pXVels[i];
                        double  yVelP = ((y == (yDim - 1)) && !wrapY) ? DisplacedY3D(pYVels, x, y + 1, z, xDim, yDim, zDim, wrapY) : pYVels[i];
                        double  zVelP = ((z == (zDim - 1)) && !wrapZ) ? DisplacedZ3D(pZVels, x, y, z + 1, xDim, yDim, zDim, wrapZ) : pZVels[i];
                        double  xVelM = DisplacedX3D(pXVels, x - 1, y, z, xDim, yDim, zDim, wrapX);
                        double  yVelM = DisplacedY3D(pYVels, x, y - 1, z, xDim, yDim, zDim, wrapY);
                        double  zVelM = DisplacedZ3D(pZVels, x, y, z - 1, xDim, yDim, zDim, wrapZ);
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
                            xFluxP = xVelP * DisplacedX3D(pField, x + 1, y, z, xDim, yDim, zDim, wrapX);
                        }
                        if (xVelM > 0)
                        {
                            xFluxM = xVelM * DisplacedX3D(pField, x - 1, y, z, xDim, yDim, zDim, wrapX);
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
                            yFluxP = yVelP * DisplacedY3D(pField, x, y + 1, z, xDim, yDim, zDim, wrapY);
                        }
                        if (yVelM > 0)
                        {
                            yFluxM = yVelM * DisplacedY3D(pField, x, y - 1, z, xDim, yDim, zDim, wrapY);
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
                            zFluxP = zVelP * DisplacedZ3D(pField, x, y, z + 1, xDim, yDim, zDim, wrapZ);
                        }
                        if (zVelM > 0)
                        {
                            zFluxM = zVelM * DisplacedZ3D(pField, x, y, z - 1, xDim, yDim, zDim, wrapZ);
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
