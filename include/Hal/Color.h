#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Hal
{
    // Colors are packed ARGB integers (0xAARRGGBB). Fractional channel
    // inputs in [0, 1] map to 0..255.

    // Fully opaque named colors, compile-time constants so they are safe to
    // use during static initialization.
    inline constexpr int32_t RED = static_cast<int32_t>(0xFFFF0000u);
    inline constexpr int32_t GREEN = static_cast<int32_t>(0xFF00FF00u);
    inline constexpr int32_t BLUE = static_cast<int32_t>(0xFF0000FFu);
    inline constexpr int32_t BLACK = static_cast<int32_t>(0xFF000000u);
    inline constexpr int32_t WHITE = static_cast<int32_t>(0xFFFFFFFFu);
    inline constexpr int32_t YELLOW = static_cast<int32_t>(0xFFFFFF00u);
    inline constexpr int32_t CYAN = static_cast<int32_t>(0xFF00FFFFu);
    inline constexpr int32_t MAGENTA = static_cast<int32_t>(0xFFFF00FFu);

    int32_t Rgb(double r,
                double g,
                double b);

    int32_t Rgba(double r,
                 double g,
                 double b,
                 double a);

    int32_t Rgb256(int32_t r,
                   int32_t g,
                   int32_t b);

    int32_t Rgba256(int32_t r,
                    int32_t g,
                    int32_t b,
                    int32_t a);

    int32_t GetRed256(int32_t color);
    int32_t GetGreen256(int32_t color);
    int32_t GetBlue256(int32_t color);
    int32_t GetAlpha256(int32_t color);

    double GetRed(int32_t color);
    double GetGreen(int32_t color);
    double GetBlue(int32_t color);
    double GetAlpha(int32_t color);

    int32_t SetRed(int32_t color,
                   double  r);

    int32_t SetGreen(int32_t color,
                     double  g);

    int32_t SetBlue(int32_t color,
                    double  b);

    int32_t SetAlpha(int32_t color,
                     double  a);

    int32_t SetRed256(int32_t color,
                      int32_t r);

    int32_t SetGreen256(int32_t color,
                        int32_t g);

    int32_t SetBlue256(int32_t color,
                       int32_t b);

    int32_t SetAlpha256(int32_t color,
                        int32_t a);

    int32_t InvertColor(int32_t color);

    // Returns a color from the d3 category20 palette based on index [0,19].
    int32_t CategoricalColor(int32_t index);

    std::vector<int32_t> CategoricalColors(int32_t startIndex,
                                           int32_t endIndex);

    int32_t HeatMapRgb(double val);
    int32_t HeatMapRbg(double val);
    int32_t HeatMapGrb(double val);
    int32_t HeatMapGbr(double val);
    int32_t HeatMapBrg(double val);
    int32_t HeatMapBgr(double val);

    int32_t HeatMapRgb(double val,
                       double min,
                       double max);

    int32_t HeatMapRbg(double val,
                       double min,
                       double max);

    int32_t HeatMapGrb(double val,
                       double min,
                       double max);

    int32_t HeatMapGbr(double val,
                       double min,
                       double max);

    int32_t HeatMapBrg(double val,
                       double min,
                       double max);

    int32_t HeatMapBgr(double val,
                       double min,
                       double max);

    int32_t HeatMapJet(double val);

    int32_t HeatMapJet(double val,
                       double min,
                       double max);

    int32_t HeatMapParula(double val);

    int32_t HeatMapParula(double val,
                          double min,
                          double max);

    int32_t ColorMap(double  val,
                     int32_t minColor,
                     int32_t maxColor);

    int32_t ColorMap(double  val,
                     double  min,
                     double  max,
                     int32_t minColor,
                     int32_t maxColor);

    int32_t ColorMap2D(double  valX,
                       double  valY,
                       int32_t bottomLeft,
                       int32_t bottomRight,
                       int32_t topLeft,
                       int32_t topRight);

    std::string ColorString(int32_t color);

    // Generates an RGB color from the YCbCr color space (y, cb, cr in [0,1]).
    int32_t YCbCrColor(double y,
                       double cb,
                       double cr);

    // Generates an RGB color from the CbCr plane at a fixed luma (x,y in [0,1]).
    int32_t CbCrPlaneColor(double x,
                           double y);

    int32_t HsLuvColor(double x,
                       double y);

    int32_t GreyScale(double val,
                      double min,
                      double max);

    int32_t GreyScale(double val);

    // HSB (a.k.a. HSV) color space. hue/saturation/brightness are in [0,1]
    // (hue wraps). Returns a packed ARGB int with full alpha.
    int32_t HsbColor(double  hue,
                     double  saturation,
                     double  brightness);

    // Writes the [hue, saturation, brightness] (each in [0,1]) of a packed
    // color into pHsbOut[0..2]. Inverse of HsbColor.
    void ColorToHsb(int32_t  color,
                    double  *pHsbOut);
}
