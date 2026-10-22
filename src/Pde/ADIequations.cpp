#include "Hal/Pde/ADIequations.h"

namespace Hal
{
    namespace Internal
    {
        double ExplicitDiffusion1ADI(int32_t       x,
                                     const double *pField,
                                     double        diffRate,
                                     int32_t       xDim,
                                     bool          wrapX)
        {
            double centerVal = pField[x];
            double valSum = Delta1D(pField, centerVal, x + 1, xDim, wrapX);
            valSum += Delta1D(pField, centerVal, x - 1, xDim, wrapX);
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
            double  deltaSum  = DeltaX2D(pField, centerVal, x + 1, y, xDim, yDim, wrapX);
            deltaSum += DeltaX2D(pField, centerVal, x - 1, y, xDim, yDim, wrapX);
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
            double  deltaSum  = DeltaY2D(pField, centerVal, x, y + 1, xDim, yDim, wrapY);
            deltaSum += DeltaY2D(pField, centerVal, x, y - 1, xDim, yDim, wrapY);
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
            double  deltaSum  = DeltaY3D(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY);
            deltaSum += DeltaY3D(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY);
            deltaSum += DeltaZ3D(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ);
            deltaSum += DeltaZ3D(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ);
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
            double  deltaSum  = DeltaX3D(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX);
            deltaSum += DeltaX3D(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX);
            deltaSum += DeltaZ3D(pField, centerVal, x, y, z + 1, xDim, yDim, zDim, wrapZ);
            deltaSum += DeltaZ3D(pField, centerVal, x, y, z - 1, xDim, yDim, zDim, wrapZ);
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
            double  deltaSum  = DeltaX3D(pField, centerVal, x + 1, y, z, xDim, yDim, zDim, wrapX);
            deltaSum += DeltaX3D(pField, centerVal, x - 1, y, z, xDim, yDim, zDim, wrapX);
            deltaSum += DeltaY3D(pField, centerVal, x, y + 1, z, xDim, yDim, zDim, wrapY);
            deltaSum += DeltaY3D(pField, centerVal, x, y - 1, z, xDim, yDim, zDim, wrapY);
            return centerVal + (deltaSum * diffRate) / 3;
        }
    }
}
