#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Hal
{
    // Packed ARGB color integers, matching HAL's Java color representation
    // (0xAARRGGBB) so ported model code and future GUI/export code (Phase 6)
    // can reuse the same on-disk/in-memory color format.

    // The named default colors from HAL's Util.java (fully opaque). Defined
    // via Rgb() so their packing is identical to a computed color.
    extern const int32_t RED;
    extern const int32_t GREEN;
    extern const int32_t BLUE;
    extern const int32_t BLACK;
    extern const int32_t WHITE;
    extern const int32_t YELLOW;
    extern const int32_t CYAN;
    extern const int32_t MAGENTA;

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

    extern const int32_t kRed;
    extern const int32_t kGreen;
    extern const int32_t kBlue;
    extern const int32_t kBlack;
    extern const int32_t kWhite;
    extern const int32_t kYellow;
    extern const int32_t kCyan;
    extern const int32_t kMagenta;

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
    // (hue wraps). Returns a packed ARGB int with full alpha. Matches the
    // conversion java.awt.Color.HSBtoRGB used in HAL's Java Util.
    int32_t HsbColor(double  hue,
                     double  saturation,
                     double  brightness);

    // Writes the [hue, saturation, brightness] (each in [0,1]) of a packed
    // color into pHsbOut[0..2]. Inverse of HsbColor; matches
    // java.awt.Color.RGBtoHSB.
    void ColorToHsb(int32_t  color,
                    double  *pHsbOut);
}
