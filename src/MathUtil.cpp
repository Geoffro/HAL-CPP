#include "Hal/MathUtil.h"

#include <cmath>

namespace Hal
{
    double ProbScale(double prob,
                     double duration)
    {
        return 1.0 - std::pow(1.0 - prob, duration);
    }

    double Bound(double val,
                 double min,
                 double max)
    {
        return (val < min) ? min : ((val > max) ? max : val);
    }

    int32_t Bound(int32_t val,
                  int32_t min,
                  int32_t max)
    {
        return (val < min) ? min : ((val > max) ? max : val);
    }

    int64_t Bound(int64_t val,
                  int64_t min,
                  int64_t max)
    {
        return (val < min) ? min : ((val > max) ? max : val);
    }

    float Bound(float val,
                float min,
                float max)
    {
        return (val < min) ? min : ((val > max) ? max : val);
    }

    int32_t Wrap(int32_t val,
                 int32_t max)
    {
        int32_t mod = val % max;
        return (mod < 0) ? (max + mod) : mod;
    }

    double Wrap(double val,
                double max)
    {
        double mod = std::fmod(val, max);
        return (mod < 0) ? (max + mod) : mod;
    }

    double ScaleMinToMax(double val,
                         double min,
                         double max)
    {
        return val * (max - min) + min;
    }

    double Scale0To1(double val,
                     double min,
                     double max)
    {
        return (val - min) / (max - min);
    }

    double Rescale(double val,
                   double oldMin,
                   double oldMax,
                   double newMin,
                   double newMax)
    {
        return ScaleMinToMax(Scale0To1(val, oldMin, oldMax), newMin, newMax);
    }

    double RescaleBounded(double val,
                          double oldMin,
                          double oldMax,
                          double newMin,
                          double newMax)
    {
        return Bound(ScaleMinToMax(Scale0To1(val, oldMin, oldMax), newMin, newMax), newMin, newMax);
    }

    double InterpolateLinear(double x,
                             double startX,
                             double endX,
                             double startY,
                             double endY)
    {
        return (startY * (endX - x) + endY * (x - startX)) / (endX - startX);
    }

    double Interpolate(double val,
                       double min,
                       double max)
    {
        val = Bound(val, 0.0, 1.0);
        return (max - min) * val + min;
    }

    double InterpolateNoBound(double val,
                              double min,
                              double max)
    {
        return (max - min) * val + min;
    }

    double Interpolate2D(double x,
                         double y,
                         double bottomLeft,
                         double bottomRight,
                         double topLeft,
                         double topRight)
    {
        x = Bound(x, 0.0, 1.0);
        y = Bound(y, 0.0, 1.0);
        double bottom = (bottomRight - bottomLeft) * x + bottomLeft;
        double top = (topRight - topLeft) * x + topLeft;
        return (top - bottom) * y + bottom;
    }

    double Interpolate2DNoBound(double x,
                                double y,
                                double bottomLeft,
                                double bottomRight,
                                double topLeft,
                                double topRight)
    {
        double bottom = (bottomRight - bottomLeft) * x + bottomLeft;
        double top = (topRight - topLeft) * x + topLeft;
        return (top - bottom) * y + bottom;
    }

    bool GetBit(int32_t v,
                int32_t i)
    {
        return ((v >> i) & 1) != 0;
    }

    bool GetBit(int64_t v,
                int32_t i)
    {
        return ((v >> i) & 1) != 0;
    }

    int32_t FlipBit(int32_t v,
                    int32_t i)
    {
        return v ^ (1 << i);
    }

    int64_t FlipBit(int64_t v,
                    int32_t i)
    {
        return v ^ (1LL << i);
    }

    int32_t SetBit(int32_t v,
                   int32_t i,
                   bool    val)
    {
        return val ? (v | (1 << i)) : (v & ~(1 << i));
    }

    int64_t SetBit(int64_t v,
                   int32_t i,
                   bool    val)
    {
        return val ? (v | (1LL << i)) : (v & ~(1LL << i));
    }

    double Dist(double x1,
                double y1,
                double x2,
                double y2)
    {
        return std::sqrt(DistSquared(x1, y1, x2, y2));
    }

    double Dist(double x1,
                double y1,
                double x2,
                double y2,
                double xDim,
                double yDim,
                bool   wrapX,
                bool   wrapY)
    {
        return std::sqrt(DistSquared(x1, y1, x2, y2, xDim, yDim, wrapX, wrapY));
    }

    double Dist(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2)
    {
        return std::sqrt(DistSquared(x1, y1, z1, x2, y2, z2));
    }

    double Dist(double  x1,
                double  y1,
                double  z1,
                double  x2,
                double  y2,
                double  z2,
                int32_t xDim,
                int32_t yDim,
                int32_t zDim,
                bool    wrapX,
                bool    wrapY,
                bool    wrapZ)
    {
        return std::sqrt(DistSquared(x1, y1, z1, x2, y2, z2, xDim, yDim, zDim, wrapX, wrapY, wrapZ));
    }

    double DistSquared(double x1,
                       double y1,
                       double x2,
                       double y2)
    {
        double xDist = x2 - x1;
        double yDist = y2 - y1;
        return xDist * xDist + yDist * yDist;
    }

    double DistSquared(double x1,
                       double y1,
                       double x2,
                       double y2,
                       double xDim,
                       double yDim,
                       bool   wrapX,
                       bool   wrapY)
    {
        double xDist = wrapX ? DispWrap(x2, x1, xDim) : (x2 - x1);
        double yDist = wrapY ? DispWrap(y2, y1, yDim) : (y2 - y1);
        return xDist * xDist + yDist * yDist;
    }

    double DistSquared(double x1,
                       double y1,
                       double z1,
                       double x2,
                       double y2,
                       double z2)
    {
        double xDist = x2 - x1;
        double yDist = y2 - y1;
        double zDist = z2 - z1;
        return xDist * xDist + yDist * yDist + zDist * zDist;
    }

    double DistSquared(double  x1,
                       double  y1,
                       double  z1,
                       double  x2,
                       double  y2,
                       double  z2,
                       int32_t xDim,
                       int32_t yDim,
                       int32_t zDim,
                       bool    wrapX,
                       bool    wrapY,
                       bool    wrapZ)
    {
        double xDist = wrapX ? DispWrap(x2, x1, xDim) : (x2 - x1);
        double yDist = wrapY ? DispWrap(y2, y1, yDim) : (y2 - y1);
        double zDist = wrapZ ? DispWrap(z2, z1, zDim) : (z2 - z1);
        return xDist * xDist + yDist * yDist + zDist * zDist;
    }

    double DistSquared(const double *pP1,
                       const double *pP2,
                       int32_t       numDims)
    {
        double sum = 0;
        for (int32_t i = 0; i < numDims; i++)
        {
            double diff = pP1[i] - pP2[i];
            sum += diff * diff;
        }
        return sum;
    }

    double Norm(double v1,
                double v2)
    {
        return std::sqrt((v1 * v1) + (v2 * v2));
    }

    double Norm(double v1,
                double v2,
                double v3)
    {
        return std::sqrt((v1 * v1) + (v2 * v2) + (v3 * v3));
    }

    double Norm(double v1,
                double v2,
                double v3,
                double v4)
    {
        return std::sqrt((v1 * v1) + (v2 * v2) + (v3 * v3) + (v4 * v4));
    }

    double Norm(const double *pVals,
                int32_t       length)
    {
        return std::sqrt(NormSquared(pVals, length));
    }

    void Normalize(double  *pVals,
                   int32_t  length)
    {
        double norm = Norm(pVals, length);
        for (int32_t i = 0; i < length; i++)
        {
            pVals[i] = pVals[i] / norm;
        }
    }

    double NormSquared(double v1,
                       double v2)
    {
        return (v1 * v1) + (v2 * v2);
    }

    double NormSquared(double v1,
                       double v2,
                       double v3)
    {
        return (v1 * v1) + (v2 * v2) + (v3 * v3);
    }

    double NormSquared(double v1,
                       double v2,
                       double v3,
                       double v4)
    {
        return (v1 * v1) + (v2 * v2) + (v3 * v3) + (v4 * v4);
    }

    double NormSquared(const double *pVals,
                       int32_t       length)
    {
        double tot = 0;
        for (int32_t i = 0; i < length; i++)
        {
            tot += pVals[i] * pVals[i];
        }
        return tot;
    }

    double DispWrap(double p1,
                    double p2,
                    double dim)
    {
        if (std::abs(p1 - p2) > (dim / 2))
        {
            if (p1 > p2)
            {
                p2 = p2 + dim;
            }
            else
            {
                p1 = p1 + dim;
            }
        }
        return p2 - p1;
    }

    bool InDim(int32_t val,
               int32_t dim)
    {
        return (val >= 0) && (val < dim);
    }

    bool InDim(double val,
               double dim)
    {
        return (val >= 0) && (val < dim);
    }
}
