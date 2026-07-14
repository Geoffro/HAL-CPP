#include "Hal/Color.h"

#include "Hal/MathUtil.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Hal
{
    int32_t Rgb(double r,
                double g,
                double b)
    {
        return Rgba(r, g, b, 1.0);
    }

    // Named default colors (see Color.h). Rgb() is a plain function, so
    // invoking it during static initialization here is well-defined
    // regardless of where it appears in this file.
    const int32_t RED = Rgb(1.0, 0.0, 0.0);
    const int32_t GREEN = Rgb(0.0, 1.0, 0.0);
    const int32_t BLUE = Rgb(0.0, 0.0, 1.0);
    const int32_t BLACK = Rgb(0.0, 0.0, 0.0);
    const int32_t WHITE = Rgb(1.0, 1.0, 1.0);
    const int32_t YELLOW = Rgb(1.0, 1.0, 0.0);
    const int32_t CYAN = Rgb(0.0, 1.0, 1.0);
    const int32_t MAGENTA = Rgb(1.0, 0.0, 1.0);

    int32_t Rgba(double r,
                 double g,
                 double b,
                 double a)
    {
        uint32_t ri = static_cast<uint32_t>(Bound(static_cast<int32_t>(r * 256), 0, 255));
        uint32_t gi = static_cast<uint32_t>(Bound(static_cast<int32_t>(g * 256), 0, 255));
        uint32_t bi = static_cast<uint32_t>(Bound(static_cast<int32_t>(b * 256), 0, 255));
        uint32_t ai = static_cast<uint32_t>(Bound(static_cast<int32_t>(a * 256), 0, 255));
        return static_cast<int32_t>((ai << 24) | (ri << 16) | (gi << 8) | bi);
    }

    int32_t Rgb256(int32_t r,
                   int32_t g,
                   int32_t b)
    {
        return Rgba256(r, g, b, 255);
    }

    int32_t Rgba256(int32_t r,
                    int32_t g,
                    int32_t b,
                    int32_t a)
    {
        uint32_t ri = static_cast<uint32_t>(Bound(r, 0, 255));
        uint32_t gi = static_cast<uint32_t>(Bound(g, 0, 255));
        uint32_t bi = static_cast<uint32_t>(Bound(b, 0, 255));
        uint32_t ai = static_cast<uint32_t>(Bound(a, 0, 255));
        return static_cast<int32_t>((ai << 24) | (ri << 16) | (gi << 8) | bi);
    }

    int32_t GetRed256(int32_t color)
    {
        return (color & 0x00ff0000) >> 16;
    }

    int32_t GetGreen256(int32_t color)
    {
        return (color & 0x0000ff00) >> 8;
    }

    int32_t GetBlue256(int32_t color)
    {
        return color & 0x000000ff;
    }

    int32_t GetAlpha256(int32_t color)
    {
        return static_cast<int32_t>((static_cast<uint32_t>(color) >> 24));
    }

    double GetRed(int32_t color)
    {
        return GetRed256(color) / 255.0;
    }

    double GetGreen(int32_t color)
    {
        return GetGreen256(color) / 255.0;
    }

    double GetBlue(int32_t color)
    {
        return GetBlue256(color) / 255.0;
    }

    double GetAlpha(int32_t color)
    {
        return GetAlpha256(color) / 255.0;
    }

    int32_t SetRed(int32_t color,
                   double  r)
    {
        return SetRed256(color, static_cast<int32_t>(Bound(r, 0.0, 1.0) * 256));
    }

    int32_t SetGreen(int32_t color,
                     double  g)
    {
        return SetGreen256(color, static_cast<int32_t>(Bound(g, 0.0, 1.0) * 256));
    }

    int32_t SetBlue(int32_t color,
                    double  b)
    {
        return SetBlue256(color, static_cast<int32_t>(Bound(b, 0.0, 1.0) * 256));
    }

    int32_t SetAlpha(int32_t color,
                     double  a)
    {
        return SetAlpha256(color, static_cast<int32_t>(Bound(a, 0.0, 1.0) * 256));
    }

    int32_t SetRed256(int32_t color,
                      int32_t r)
    {
        r = Bound(r, 0, 255);
        return (color & static_cast<int32_t>(0xff00ffff)) | (r << 16);
    }

    int32_t SetGreen256(int32_t color,
                        int32_t g)
    {
        g = Bound(g, 0, 255);
        return (color & static_cast<int32_t>(0xffff00ff)) | (g << 8);
    }

    int32_t SetBlue256(int32_t color,
                       int32_t b)
    {
        b = Bound(b, 0, 255);
        return (color & static_cast<int32_t>(0xffffff00)) | b;
    }

    int32_t SetAlpha256(int32_t color,
                        int32_t a)
    {
        a = Bound(a, 0, 255);
        return static_cast<int32_t>((color & 0x00ffffff) | (static_cast<uint32_t>(a) << 24));
    }

    int32_t InvertColor(int32_t color)
    {
        return Rgb256(255 - GetRed256(color), 255 - GetGreen256(color), 255 - GetBlue256(color));
    }

    const int32_t kRed = Rgb(1, 0, 0);
    const int32_t kGreen = Rgb(0, 1, 0);
    const int32_t kBlue = Rgb(0, 0, 1);
    const int32_t kBlack = Rgb(0, 0, 0);
    const int32_t kWhite = Rgb(1, 1, 1);
    const int32_t kYellow = Rgb(1, 1, 0);
    const int32_t kCyan = Rgb(0, 1, 1);
    const int32_t kMagenta = Rgb(1, 0, 1);

    namespace Internal
    {
        const int32_t kCategoricalColorTable[20] =
            {
                Rgb256(56, 116, 177),
                Rgb256(198, 56, 44),
                Rgb256(79, 159, 57),
                Rgb256(189, 190, 58),
                Rgb256(142, 102, 186),
                Rgb256(240, 134, 39),
                Rgb256(83, 187, 206),
                Rgb256(214, 123, 191),
                Rgb256(133, 88, 76),
                Rgb256(178, 197, 230),
                Rgb256(243, 156, 151),
                Rgb256(166, 222, 144),
                Rgb256(220, 220, 147),
                Rgb256(194, 174, 211),
                Rgb256(246, 191, 126),
                Rgb256(169, 216, 228),
                Rgb256(238, 184, 209),
                Rgb256(190, 157, 146),
                Rgb256(199, 199, 199),
                Rgb256(127, 127, 127)};

        int32_t InterpComp(double  val,
                           int32_t minComp,
                           int32_t maxComp)
        {
            return static_cast<int32_t>((maxComp - minComp) * val) + minComp;
        }
    }

    int32_t CategoricalColor(int32_t index)
    {
        if ((index < 0) || (index > 19))
        {
            throw std::invalid_argument("index outside color category range [0-19]");
        }
        return Internal::kCategoricalColorTable[index];
    }

    std::vector<int32_t> CategoricalColors(int32_t startIndex,
                                           int32_t endIndex)
    {
        std::vector<int32_t> ret(endIndex - startIndex);
        for (int32_t i = startIndex; i < endIndex; i++)
        {
            ret[i - startIndex] = CategoricalColor(i);
        }
        return ret;
    }

    int32_t HeatMapRgb(double val)
    {
        double c1 = val * 4;
        double c2 = (val - 0.25) * 2;
        double c3 = (val - 0.75) * 4;
        return Rgb(c1, c2, c3);
    }

    int32_t HeatMapRbg(double val)
    {
        double c1 = val * 4;
        double c2 = (val - 0.25) * 2;
        double c3 = (val - 0.75) * 4;
        return Rgb(c1, c3, c2);
    }

    int32_t HeatMapGrb(double val)
    {
        double c1 = val * 4;
        double c2 = (val - 0.25) * 2;
        double c3 = (val - 0.75) * 4;
        return Rgb(c2, c1, c3);
    }

    int32_t HeatMapGbr(double val)
    {
        double c1 = val * 4;
        double c2 = (val - 0.25) * 2;
        double c3 = (val - 0.75) * 4;
        return Rgb(c3, c1, c2);
    }

    int32_t HeatMapBrg(double val)
    {
        double c1 = val * 4;
        double c2 = (val - 0.25) * 2;
        double c3 = (val - 0.75) * 4;
        return Rgb(c2, c3, c1);
    }

    int32_t HeatMapBgr(double val)
    {
        double c1 = val * 4;
        double c2 = (val - 0.25) * 2;
        double c3 = (val - 0.75) * 4;
        return Rgb(c3, c2, c1);
    }

    int32_t HeatMapRgb(double val,
                       double min,
                       double max)
    {
        return HeatMapRgb(Scale0To1(val, min, max));
    }

    // NOTE: HAL's Java HeatMapRBG(val,min,max) called HeatMapBGR(val)
    // instead of HeatMapRBG(val) — every sibling overload calls its own
    // matching single-argument palette function, so this was a copy-paste
    // bug (see docs/JAVA_BUGS_FOUND.md #3); fixed here.
    int32_t HeatMapRbg(double val,
                       double min,
                       double max)
    {
        return HeatMapRbg(Scale0To1(val, min, max));
    }

    int32_t HeatMapGrb(double val,
                       double min,
                       double max)
    {
        return HeatMapGrb(Scale0To1(val, min, max));
    }

    int32_t HeatMapGbr(double val,
                       double min,
                       double max)
    {
        return HeatMapGbr(Scale0To1(val, min, max));
    }

    int32_t HeatMapBrg(double val,
                       double min,
                       double max)
    {
        return HeatMapBrg(Scale0To1(val, min, max));
    }

    int32_t HeatMapBgr(double val,
                       double min,
                       double max)
    {
        return HeatMapBgr(Scale0To1(val, min, max));
    }

    int32_t HeatMapJet(double val)
    {
        if (val <= 0)
        {
            return Rgb(0, 0, 0.5);
        }
        if (val >= 1)
        {
            return Rgb(0.5, 0, 0);
        }
        double c1 = 1.5 - std::abs(0.75 - val) * 4;
        double c2 = 1.5 - std::abs(0.5 - val) * 4;
        double c3 = 1.5 - std::abs(0.25 - val) * 4;
        return Rgb(c1, c2, c3);
    }

    int32_t HeatMapJet(double val,
                       double min,
                       double max)
    {
        return HeatMapJet(Scale0To1(val, min, max));
    }

    int32_t HeatMapParula(double val)
    {
        if (val <= 0)
        {
            return Rgb(0.2, 0.2, 0.5);
        }
        if (val >= 1)
        {
            return Rgb(1.0, 1.0, 0.0);
        }
        if (val < 0.125)
        {
            return Rgb(0.2 - val * 1.6, 0.2 + val * 0.8, 0.5 + val * 2);
        }
        return Rgb((val - 0.375) * 2, 0.2 + val * 0.8, 0.9 - val * 1.143);
    }

    int32_t HeatMapParula(double val,
                          double min,
                          double max)
    {
        return HeatMapParula(Scale0To1(val, min, max));
    }

    int32_t ColorMap(double  val,
                     int32_t minColor,
                     int32_t maxColor)
    {
        return ColorMap(val, 0, 1, minColor, maxColor);
    }

    int32_t ColorMap(double  val,
                     double  min,
                     double  max,
                     int32_t minColor,
                     int32_t maxColor)
    {
        if (val <= min)
        {
            return minColor;
        }
        if (val >= max)
        {
            return maxColor;
        }
        val = (val - min) / (max - min);
        return Rgb256(
            Internal::InterpComp(val, GetRed256(minColor), GetRed256(maxColor)),
            Internal::InterpComp(val, GetGreen256(minColor), GetGreen256(maxColor)),
            Internal::InterpComp(val, GetBlue256(minColor), GetBlue256(maxColor)));
    }

    int32_t ColorMap2D(double  valX,
                       double  valY,
                       int32_t bottomLeft,
                       int32_t bottomRight,
                       int32_t topLeft,
                       int32_t topRight)
    {
        return Rgb(
            Interpolate2D(valX, valY, GetRed(bottomLeft), GetRed(bottomRight), GetRed(topLeft), GetRed(topRight)),
            Interpolate2D(valX, valY, GetGreen(bottomLeft), GetGreen(bottomRight), GetGreen(topLeft), GetGreen(topRight)),
            Interpolate2D(valX, valY, GetBlue(bottomLeft), GetBlue(bottomRight), GetBlue(topLeft), GetBlue(topRight)));
    }

    std::string ColorString(int32_t color)
    {
        return "r: " + std::to_string(GetRed256(color)) +
               ", g: " + std::to_string(GetGreen256(color)) +
               ", b: " + std::to_string(GetBlue256(color)) +
               ", a: " + std::to_string(GetAlpha256(color));
    }

    int32_t YCbCrColor(double y,
                       double cb,
                       double cr)
    {
        double yd = Bound(y * 256, 0.0, 255.0);
        double cbd = Bound(cb * 256, 0.0, 255.0);
        double crd = Bound(cr * 256, 0.0, 255.0);
        int32_t r = static_cast<int32_t>(yd + 1.402 * (crd - 128));
        int32_t g = static_cast<int32_t>(yd - 0.344136 * (cbd - 128) - 0.714136 * (crd - 128));
        int32_t b = static_cast<int32_t>(yd + 1.772 * (cbd - 128));
        return Rgb256(r, g, b);
    }

    int32_t CbCrPlaneColor(double x,
                           double y)
    {
        const double ycbcrY = 0.5;
        return YCbCrColor(ycbcrY, x, y);
    }

    int32_t HsLuvColor(double x,
                       double y)
    {
        double xScaled = Bound(y, 0.0, 0.99999999) * 2;
        double yScaled = (0.99999999 - Bound(x, 0.0, 0.99999999)) * 2;
        int32_t xInt = static_cast<int32_t>(xScaled);
        int32_t yInt = static_cast<int32_t>(yScaled);
        xScaled = xScaled - xInt;
        yScaled = yScaled - yInt;
        double r = 0;
        double g = 0;
        double b = 0;
        if ((xInt == 0) && (yInt == 0))
        {
            // bottom left
            r = Interpolate2D(xScaled, yScaled, 89, 200, 60, 158);
            g = Interpolate2D(xScaled, yScaled, 60, 60, 199, 157);
            b = Interpolate2D(xScaled, yScaled, 255, 255, 255, 158);
        }
        else if ((xInt == 0) && (yInt == 1))
        {
            // top left
            r = Interpolate2D(xScaled, yScaled, 60, 158, 60, 255);
            g = Interpolate2D(xScaled, yScaled, 199, 157, 255, 247);
            b = Interpolate2D(xScaled, yScaled, 255, 158, 85, 60);
        }
        else if ((xInt == 1) && (yInt == 0))
        {
            // bottom right
            r = Interpolate2D(xScaled, yScaled, 200, 255, 158, 255);
            g = Interpolate2D(xScaled, yScaled, 60, 60, 157, 60);
            b = Interpolate2D(xScaled, yScaled, 255, 230, 158, 106);
        }
        else if ((xInt == 1) && (yInt == 1))
        {
            // top right
            r = Interpolate2D(xScaled, yScaled, 158, 255, 255, 255);
            g = Interpolate2D(xScaled, yScaled, 157, 60, 247, 132);
            b = Interpolate2D(xScaled, yScaled, 158, 106, 60, 60);
        }
        return Rgb256(static_cast<int32_t>(r), static_cast<int32_t>(g), static_cast<int32_t>(b));
    }

    int32_t GreyScale(double val,
                      double min,
                      double max)
    {
        double colorVal = (val - min) / (max - min);
        return Rgb(colorVal, colorVal, colorVal);
    }

    int32_t GreyScale(double val)
    {
        return Rgb(val, val, val);
    }

    int32_t HsbColor(double  hue,
                     double  saturation,
                     double  brightness)
    {
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        if (saturation == 0.0)
        {
            r = brightness;
            g = brightness;
            b = brightness;
        }
        else
        {
            double h = (hue - std::floor(hue)) * 6.0;
            double f = h - std::floor(h);
            double p = brightness * (1.0 - saturation);
            double q = brightness * (1.0 - saturation * f);
            double t = brightness * (1.0 - saturation * (1.0 - f));
            switch (static_cast<int32_t>(h))
            {
                case 0:
                    r = brightness;
                    g = t;
                    b = p;
                    break;
                case 1:
                    r = q;
                    g = brightness;
                    b = p;
                    break;
                case 2:
                    r = p;
                    g = brightness;
                    b = t;
                    break;
                case 3:
                    r = p;
                    g = q;
                    b = brightness;
                    break;
                case 4:
                    r = t;
                    g = p;
                    b = brightness;
                    break;
                default:
                    r = brightness;
                    g = p;
                    b = q;
                    break;
            }
        }
        return Rgb256(static_cast<int32_t>(r * 255.0 + 0.5), static_cast<int32_t>(g * 255.0 + 0.5), static_cast<int32_t>(b * 255.0 + 0.5));
    }

    void ColorToHsb(int32_t  color,
                    double  *pHsbOut)
    {
        int32_t r = GetRed256(color);
        int32_t g = GetGreen256(color);
        int32_t b = GetBlue256(color);
        int32_t cMax = std::max(r, std::max(g, b));
        int32_t cMin = std::min(r, std::min(g, b));

        double brightness = cMax / 255.0;
        double saturation = (cMax != 0) ? (static_cast<double>(cMax - cMin) / cMax) : 0.0;
        double hue = 0.0;
        if (saturation != 0.0)
        {
            double range = static_cast<double>(cMax - cMin);
            double redComp = (cMax - r) / range;
            double greenComp = (cMax - g) / range;
            double blueComp = (cMax - b) / range;
            if (r == cMax)
            {
                hue = blueComp - greenComp;
            }
            else if (g == cMax)
            {
                hue = 2.0 + redComp - blueComp;
            }
            else
            {
                hue = 4.0 + greenComp - redComp;
            }
            hue /= 6.0;
            if (hue < 0.0)
            {
                hue += 1.0;
            }
        }
        pHsbOut[0] = hue;
        pHsbOut[1] = saturation;
        pHsbOut[2] = brightness;
    }
}
