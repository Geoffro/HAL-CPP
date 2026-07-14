#include "Hal/Pde/ADIequations.h"

namespace Hal
{
    namespace Internal
    {
        double Delta1DADI(const double *pField,
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

        double DeltaX2DADI(const double *pField,
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

        double DeltaY2DADI(const double *pField,
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

        double DeltaX3DADI(const double *pField,
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

        double DeltaY3DADI(const double *pField,
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

        double DeltaZ3DADI(const double *pField,
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

        double ExplicitDiffusion1ADI(int32_t       x,
                                     const double *pField,
                                     double        diffRate,
                                     int32_t       xDim,
                                     bool          wrapX)
        {
            double centerVal = pField[x];
            double valSum = Delta1DADI(pField, centerVal, x + 1, xDim, wrapX);
            valSum += Delta1DADI(pField, centerVal, x - 1, xDim, wrapX);
            return centerVal + (valSum * diffRate) / 2;
        }

        double ExplicitDiffusionX2ADI(int32_t       x,
                                      int32_t       y,
                                      const double *pField,
                                      double        diffRate,
                                      int32_t       xDim,
                                      int32_t       yDim,
                                      bool          wrapX)
        {
            int32_t i = x * yDim + y;
            double  centerVal = pField[i];
            double  deltaSum  = DeltaX2DADI(pField, centerVal, x + 1, y, xDim, yDim, wrapX);
            deltaSum += DeltaX2DADI(pField, centerVal, x - 1, y, xDim, yDim, wrapX);
            return centerVal + (deltaSum * diffRate) / 2;
        }

        double ExplicitDiffusionY2ADI(int32_t       x,
                                      int32_t       y,
                                      const double *pField,
                                      double        diffRate,
                                      int32_t       xDim,
                                      int32_t       yDim,
                                      bool          wrapY)
        {
            int32_t i = x * yDim + y;
            double  centerVal = pField[i];
            double  deltaSum  = DeltaY2DADI(pField, centerVal, x, y + 1, xDim, yDim, wrapY);
            deltaSum += DeltaY2DADI(pField, centerVal, x, y - 1, xDim, yDim, wrapY);
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
                                       bool          wrapZ)
        {
            int32_t i = x * yDim * zDim + y * zDim + z;
            double  centerVal = pField[i];
            double  deltaSum  = DeltaY3DADI(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY);
            deltaSum += DeltaY3DADI(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY);
            deltaSum += DeltaZ3DADI(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ);
            deltaSum += DeltaZ3DADI(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ);
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
                                       bool          wrapZ)
        {
            int32_t i = x * yDim * zDim + y * zDim + z;
            double  centerVal = pField[i];
            double  deltaSum  = DeltaX3DADI(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX);
            deltaSum += DeltaX3DADI(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX);
            deltaSum += DeltaZ3DADI(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ);
            deltaSum += DeltaZ3DADI(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ);
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
                                       bool          wrapY)
        {
            int32_t i = x * yDim * zDim + y * zDim + z;
            double  centerVal = pField[i];
            double  deltaSum  = DeltaX3DADI(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX);
            deltaSum += DeltaX3DADI(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX);
            deltaSum += DeltaY3DADI(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY);
            deltaSum += DeltaY3DADI(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY);
            return centerVal + (deltaSum * diffRate) / 3;
        }
    }
}
