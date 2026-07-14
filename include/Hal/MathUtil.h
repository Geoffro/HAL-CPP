#pragma once

#include <cstdint>

namespace Hal
{
    double Bound(double val,
                 double min,
                 double max);

    int32_t Bound(int32_t val,
                  int32_t min,
                  int32_t max);

    int64_t Bound(int64_t val,
                  int64_t min,
                  int64_t max);

    float Bound(float val,
                float min,
                float max);

    // Wraps val into [0, max) with the same sign convention as a true
    // modulo (unlike C++'s built-in % on negative operands).
    int32_t Wrap(int32_t val,
                 int32_t max);

    double Wrap(double val,
                double max);

    double ScaleMinToMax(double val,
                         double min,
                         double max);

    double Scale0To1(double val,
                     double min,
                     double max);

    double Rescale(double val,
                   double oldMin,
                   double oldMax,
                   double newMin,
                   double newMax);

    double RescaleBounded(double val,
                          double oldMin,
                          double oldMax,
                          double newMin,
                          double newMax);

    double InterpolateLinear(double x,
                             double startX,
                             double endX,
                             double startY,
                             double endY);

    // Rescale a per-unit-time probability to a probability over `duration`
    // units, assuming independent trials: 1 - (1 - prob)^duration. From
    // HAL's Util.ProbScale, used to convert per-day rates to per-timestep.
    double ProbScale(double prob,
                     double duration);

    double Interpolate(double val,
                       double min,
                       double max);

    double InterpolateNoBound(double val,
                              double min,
                              double max);

    double Interpolate2D(double x,
                         double y,
                         double bottomLeft,
                         double bottomRight,
                         double topLeft,
                         double topRight);

    double Interpolate2DNoBound(double x,
                                double y,
                                double bottomLeft,
                                double bottomRight,
                                double topLeft,
                                double topRight);

    // NOTE: HAL's Java GetBit had an operator-precedence bug
    // (`v >> i % 1 != 0`, i.e. `v >> (i % 1)`) that made it ignore `i`
    // entirely (see docs/JAVA_BUGS_FOUND.md #1). Implemented correctly
    // here as `((v >> i) & 1) != 0`.
    bool GetBit(int32_t v,
                int32_t i);

    bool GetBit(int64_t v,
                int32_t i);

    int32_t FlipBit(int32_t v,
                    int32_t i);

    // NOTE: uses a 64-bit shift (`1LL << i`) rather than the Java
    // version's `1 << i`, which silently truncates for i >= 32
    // (see docs/JAVA_BUGS_FOUND.md #2).
    int64_t FlipBit(int64_t v,
                    int32_t i);

    int32_t SetBit(int32_t v,
                   int32_t i,
                   bool    val);

    int64_t SetBit(int64_t v,
                   int32_t i,
                   bool    val);

    double Dist(double x1,
                double y1,
                double x2,
                double y2);

    double Dist(double x1,
                double y1,
                double x2,
                double y2,
                double xDim,
                double yDim,
                bool   wrapX,
                bool   wrapY);

    double Dist(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2);

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
                bool    wrapZ);

    double DistSquared(double x1,
                       double y1,
                       double x2,
                       double y2);

    double DistSquared(double x1,
                       double y1,
                       double x2,
                       double y2,
                       double xDim,
                       double yDim,
                       bool   wrapX,
                       bool   wrapY);

    double DistSquared(double x1,
                       double y1,
                       double z1,
                       double x2,
                       double y2,
                       double z2);

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
                       bool    wrapZ);

    double DistSquared(const double *p1,
                       const double *p2,
                       int32_t       numDims);

    double Norm(double v1,
                double v2);

    double Norm(double v1,
                double v2,
                double v3);

    double Norm(double v1,
                double v2,
                double v3,
                double v4);

    double Norm(const double *pVals,
                int32_t       length);

    void Normalize(double  *pVals,
                   int32_t  length);

    double NormSquared(double v1,
                       double v2);

    double NormSquared(double v1,
                       double v2,
                       double v3);

    double NormSquared(double v1,
                       double v2,
                       double v3,
                       double v4);

    double NormSquared(const double *pVals,
                       int32_t       length);

    // Returns the minimum distance between p1 and p2 along a single axis
    // of length dim, using wraparound if that is shorter than the direct
    // distance.
    double DispWrap(double p1,
                    double p2,
                    double dim);

    // Returns whether val is in [0, dim).
    bool InDim(int32_t val,
               int32_t dim);

    bool InDim(double val,
               double dim);
}
