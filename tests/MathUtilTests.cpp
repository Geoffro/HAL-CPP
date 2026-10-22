#include <doctest/doctest.h>

#include "Hal/MathUtil.h"

#include <cstdint>

TEST_CASE("Bound clamps values into the inclusive [min, max] range")
{
    CHECK(Hal::Bound(5, 0, 10) == 5);
    CHECK(Hal::Bound(-5, 0, 10) == 0);
    CHECK(Hal::Bound(15, 0, 10) == 10);
    CHECK(Hal::Bound(2.5, 0.0, 1.0) == doctest::Approx(1.0));
}

TEST_CASE("Wrap keeps values within [0, max) with a true-modulo sign convention")
{
    CHECK(Hal::Wrap(5, 4) == 1);
    CHECK(Hal::Wrap(-1, 4) == 3);
    CHECK(Hal::Wrap(-5, 4) == 3);
    CHECK(Hal::Wrap(0, 4) == 0);

    CHECK(Hal::Wrap(5.5, 4.0) == doctest::Approx(1.5));
    CHECK(Hal::Wrap(-0.5, 4.0) == doctest::Approx(3.5));
}

TEST_CASE("Scale0To1 / ScaleMinToMax / Rescale round-trip correctly")
{
    CHECK(Hal::Scale0To1(5.0, 0.0, 10.0) == doctest::Approx(0.5));
    CHECK(Hal::ScaleMinToMax(0.5, 0.0, 10.0) == doctest::Approx(5.0));
    CHECK(Hal::Rescale(5.0, 0.0, 10.0, 0.0, 100.0) == doctest::Approx(50.0));
    CHECK(Hal::RescaleBounded(20.0, 0.0, 10.0, 0.0, 100.0) == doctest::Approx(100.0));
}

TEST_CASE("Interpolate variants blend correctly and bound to [0,1] input")
{
    CHECK(Hal::Interpolate(0.5, 10.0, 20.0) == doctest::Approx(15.0));
    CHECK(Hal::Interpolate(2.0, 10.0, 20.0) == doctest::Approx(20.0));
    CHECK(Hal::InterpolateNoBound(2.0, 10.0, 20.0) == doctest::Approx(30.0));
    CHECK(Hal::Interpolate2D(0.5, 0.5, 0.0, 10.0, 0.0, 10.0) == doctest::Approx(5.0));
}

TEST_CASE("GetBit/SetBit/FlipBit operate on the correct bit index")
{
    int32_t v = 0;
    CHECK_FALSE(Hal::GetBit(v, 3));

    v = Hal::SetBit(v, 3, true);
    CHECK(Hal::GetBit(v, 3));
    CHECK_FALSE(Hal::GetBit(v, 2));

    v = Hal::FlipBit(v, 3);
    CHECK_FALSE(Hal::GetBit(v, 3));

    int64_t vLong = 0;
    vLong = Hal::SetBit(vLong, 40, true);
    CHECK(Hal::GetBit(vLong, 40));
    CHECK_FALSE(Hal::GetBit(vLong, 39));
}

TEST_CASE("Dist/DistSquared match direct computation, with and without wraparound")
{
    CHECK(Hal::DistSquared(0.0, 0.0, 3.0, 4.0) == doctest::Approx(25.0));
    CHECK(Hal::Dist(0.0, 0.0, 3.0, 4.0) == doctest::Approx(5.0));

    // Wraparound: 1 and 9 on a dim-10 axis are 2 apart the short way.
    CHECK(Hal::DistSquared(1.0, 0.0, 9.0, 0.0, 10.0, 10.0, true, false) == doctest::Approx(4.0));
}

TEST_CASE("DispWrap returns the shorter signed displacement (p2 - p1) across a wrapped axis")
{
    // 1 and 9 on a dim-10 axis are 2 apart the short way, wrapping through 0.
    CHECK(Hal::DispWrap(9.0, 1.0, 10.0) == doctest::Approx(2.0));
    CHECK(Hal::DispWrap(1.0, 9.0, 10.0) == doctest::Approx(-2.0));
}

TEST_CASE("ProbScale converts a per-unit probability to a longer duration")
{
    // Over one unit of duration, the probability is unchanged.
    CHECK(Hal::ProbScale(0.3, 1.0) == doctest::Approx(0.3));
    // Over zero duration, nothing happens.
    CHECK(Hal::ProbScale(0.3, 0.0) == doctest::Approx(0.0));
    // 1 - (1-p)^d for a known case.
    CHECK(Hal::ProbScale(0.5, 2.0) == doctest::Approx(0.75));
    // A certain event stays certain.
    CHECK(Hal::ProbScale(1.0, 5.0) == doctest::Approx(1.0));
}

TEST_CASE("Bit helpers handle the sign bit")
{
    CHECK(Hal::GetBit(Hal::SetBit(static_cast<int32_t>(0), 31, true), 31));
    CHECK(Hal::FlipBit(static_cast<int32_t>(0), 31) == INT32_MIN);
    CHECK(Hal::SetBit(static_cast<int32_t>(-1), 31, false) == INT32_MAX);
    CHECK(Hal::FlipBit(static_cast<int64_t>(0), 63) == INT64_MIN);
    CHECK(Hal::SetBit(static_cast<int64_t>(-1), 63, false) == INT64_MAX);
}

TEST_CASE("InterpolateLinear, Interpolate2DNoBound, and InDim")
{
    CHECK(Hal::InterpolateLinear(15.0, 10.0, 20.0, 100.0, 200.0) == doctest::Approx(150.0));
    CHECK(Hal::Interpolate2DNoBound(0.5, 0.5, 0.0, 2.0, 4.0, 6.0) == doctest::Approx(3.0));
    CHECK(Hal::Interpolate2DNoBound(2.0, 0.0, 0.0, 1.0, 0.0, 1.0) == doctest::Approx(2.0));
    CHECK(Hal::InDim(0, 5));
    CHECK_FALSE(Hal::InDim(5, 5));
    CHECK_FALSE(Hal::InDim(-0.1, 5.0));
    CHECK(Hal::InDim(4.9, 5.0));
}

TEST_CASE("Normalize scales a vector to unit length and NormSquared matches Norm")
{
    double v[3] = {3.0, 0.0, 4.0};
    CHECK(Hal::NormSquared(v, 3) == doctest::Approx(25.0));
    CHECK(Hal::NormSquared(3.0, 4.0) == doctest::Approx(25.0));
    CHECK(Hal::NormSquared(1.0, 2.0, 2.0) == doctest::Approx(9.0));
    CHECK(Hal::NormSquared(1.0, 1.0, 1.0, 1.0) == doctest::Approx(4.0));
    Hal::Normalize(v, 3);
    CHECK(Hal::Norm(v, 3) == doctest::Approx(1.0));
    CHECK(v[2] == doctest::Approx(0.8));
}
