#include <doctest/doctest.h>

#include "Hal/Color.h"

#include <stdexcept>

TEST_CASE("Rgb256 packs and unpacks channels round-trip")
{
    int32_t color = Hal::Rgb256(10, 20, 30);
    CHECK(Hal::GetRed256(color) == 10);
    CHECK(Hal::GetGreen256(color) == 20);
    CHECK(Hal::GetBlue256(color) == 30);
    CHECK(Hal::GetAlpha256(color) == 255);
}

TEST_CASE("Rgba256 preserves the alpha channel")
{
    int32_t color = Hal::Rgba256(10, 20, 30, 128);
    CHECK(Hal::GetAlpha256(color) == 128);
}

TEST_CASE("SetRed256/SetGreen256/SetBlue256/SetAlpha256 only change their own channel")
{
    int32_t color = Hal::Rgba256(10, 20, 30, 40);
    int32_t withRed = Hal::SetRed256(color, 200);
    CHECK(Hal::GetRed256(withRed) == 200);
    CHECK(Hal::GetGreen256(withRed) == 20);
    CHECK(Hal::GetBlue256(withRed) == 30);
    CHECK(Hal::GetAlpha256(withRed) == 40);
}

TEST_CASE("InvertColor flips each channel to 255 minus its value")
{
    int32_t color = Hal::Rgb256(0, 100, 255);
    int32_t inverted = Hal::InvertColor(color);
    CHECK(Hal::GetRed256(inverted) == 255);
    CHECK(Hal::GetGreen256(inverted) == 155);
    CHECK(Hal::GetBlue256(inverted) == 0);
}

TEST_CASE("CategoricalColor rejects indices outside [0,19]")
{
    CHECK_NOTHROW(Hal::CategoricalColor(0));
    CHECK_NOTHROW(Hal::CategoricalColor(19));
    CHECK_THROWS_AS(Hal::CategoricalColor(-1), std::invalid_argument);
    CHECK_THROWS_AS(Hal::CategoricalColor(20), std::invalid_argument);
}

TEST_CASE("CategoricalColors returns one color per index in the requested range")
{
    std::vector<int32_t> colors = Hal::CategoricalColors(2, 5);
    REQUIRE(colors.size() == 3);
    CHECK(colors[0] == Hal::CategoricalColor(2));
    CHECK(colors[2] == Hal::CategoricalColor(4));
}

TEST_CASE("HeatMapJet clamps to its endpoint colors outside [0,1]")
{
    CHECK(Hal::HeatMapJet(-1.0) == Hal::Rgb(0, 0, 0.5));
    CHECK(Hal::HeatMapJet(2.0) == Hal::Rgb(0.5, 0, 0));
}

TEST_CASE("ColorMap clamps to endpoint colors outside [min,max] and interpolates within")
{
    int32_t lo = Hal::Rgb256(0, 0, 0);
    int32_t hi = Hal::Rgb256(200, 100, 50);
    CHECK(Hal::ColorMap(-1.0, 0.0, 1.0, lo, hi) == lo);
    CHECK(Hal::ColorMap(2.0, 0.0, 1.0, lo, hi) == hi);

    int32_t mid = Hal::ColorMap(0.5, 0.0, 1.0, lo, hi);
    CHECK(Hal::GetRed256(mid) == 100);
}

TEST_CASE("GreyScale produces equal R, G, and B channels")
{
    int32_t color = Hal::GreyScale(0.4);
    CHECK(Hal::GetRed256(color) == Hal::GetGreen256(color));
    CHECK(Hal::GetGreen256(color) == Hal::GetBlue256(color));
}

TEST_CASE("HsbColor produces the expected primaries")
{
    // hue 0 = red, 1/3 = green, 2/3 = blue at full saturation/brightness.
    CHECK(Hal::HsbColor(0.0, 1.0, 1.0) == Hal::Rgb256(255, 0, 0));
    CHECK(Hal::HsbColor(1.0 / 3.0, 1.0, 1.0) == Hal::Rgb256(0, 255, 0));
    CHECK(Hal::HsbColor(2.0 / 3.0, 1.0, 1.0) == Hal::Rgb256(0, 0, 255));
    // Zero saturation -> grey scaled by brightness.
    CHECK(Hal::HsbColor(0.5, 0.0, 0.5) == Hal::Rgb256(128, 128, 128));
}

TEST_CASE("ColorToHsb inverts HsbColor (round-trip through the RGB quantization)")
{
    for (double hue : {0.05, 0.2, 0.45, 0.7, 0.95})
    {
        for (double sat : {0.3, 0.7, 1.0})
        {
            for (double bright : {0.4, 0.8, 1.0})
            {
                int32_t color = Hal::HsbColor(hue, sat, bright);
                double hsb[3];
                Hal::ColorToHsb(color, hsb);
                // Re-encoding the recovered HSB must reproduce the same
                // packed color exactly (HSB->RGB is many-to-one through the
                // 8-bit quantization, so compare the round-tripped color,
                // not the raw HSB floats).
                CHECK(Hal::HsbColor(hsb[0], hsb[1], hsb[2]) == color);
            }
        }
    }
}

TEST_CASE("ColorToHsb reports brightness and saturation for known colors")
{
    double hsb[3];
    Hal::ColorToHsb(Hal::Rgb256(255, 0, 0), hsb);
    CHECK(hsb[0] == doctest::Approx(0.0));
    CHECK(hsb[1] == doctest::Approx(1.0));
    CHECK(hsb[2] == doctest::Approx(1.0));

    Hal::ColorToHsb(Hal::Rgb256(0, 0, 0), hsb);
    CHECK(hsb[1] == doctest::Approx(0.0));
    CHECK(hsb[2] == doctest::Approx(0.0));
}

TEST_CASE("Named default colors match their Rgb() equivalents and are fully opaque")
{
    CHECK(Hal::RED == Hal::Rgb(1.0, 0.0, 0.0));
    CHECK(Hal::GREEN == Hal::Rgb(0.0, 1.0, 0.0));
    CHECK(Hal::BLUE == Hal::Rgb(0.0, 0.0, 1.0));
    CHECK(Hal::BLACK == Hal::Rgb(0.0, 0.0, 0.0));
    CHECK(Hal::WHITE == Hal::Rgb(1.0, 1.0, 1.0));
    CHECK(Hal::YELLOW == Hal::Rgb(1.0, 1.0, 0.0));
    CHECK(Hal::CYAN == Hal::Rgb(0.0, 1.0, 1.0));
    CHECK(Hal::MAGENTA == Hal::Rgb(1.0, 0.0, 1.0));

    // Channel spot-checks.
    CHECK(Hal::GetRed256(Hal::RED) == 255);
    CHECK(Hal::GetGreen256(Hal::RED) == 0);
    CHECK(Hal::GetAlpha256(Hal::WHITE) == 255);
    CHECK(Hal::GetBlue256(Hal::MAGENTA) == 255);
}
