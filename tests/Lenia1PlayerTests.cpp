#include <doctest/doctest.h>

#include "Hal/Tools/Lenia1Player.h"

#include <cmath>
#include <cstdint>
#include <vector>

// A kernel that is a single point at the focal cell (distance 0): once
// normalized this is a Dirac delta, so convolving with it is the identity.
static double DeltaKernel(double distance)
{
    return (distance < 0.5) ? 1.0 : 0.0;
}

// A flat kernel over a small radius — a non-trivial averaging kernel.
static double DiscKernel(double distance)
{
    return (distance <= 3.0) ? 1.0 : 0.0;
}

TEST_CASE("Hal::Lenia1Player: side length is 2^sideLenExp and starts at tick 0")
{
    Hal::Lenia1Player player(5, 0.1, DiscKernel, [](double /*potential*/)
    {
        return 0.0;
    });
    CHECK(player.Xdim() == 32);
    CHECK(player.Ydim() == 32);
    CHECK(player.Length() == 32 * 32);
    CHECK(player.GetTick() == 0);
}

TEST_CASE("Hal::Lenia1Player: convolving a constant field with a normalized kernel leaves it constant")
{
    // Kernel sums to 1 after normalization, so K * (constant c) == c
    // everywhere — this checks both the normalization and the convolution.
    Hal::Lenia1Player player(4, 1.0, DiscKernel, [](double /*potential*/)
    {
        return 0.0;
    });
    for (int32_t i = 0; i < player.Length(); i++)
    {
        player.Set(i, 0.5);
    }

    player.Update();

    for (int32_t i = 0; i < player.Length(); i++)
    {
        CAPTURE(i);
        CHECK(player.GetConvolvedFieldVal(i) == doctest::Approx(0.5).epsilon(1e-9));
    }
}

TEST_CASE("Hal::Lenia1Player: a delta kernel makes the convolution the identity")
{
    // With a delta-at-center kernel, the shifted-kernel convolution must
    // reproduce the input field exactly — pinning down the fftshift/kernel
    // coordinate alignment. Growth returns 0 so the field itself is
    // unchanged and we can read the convolution back cleanly.
    Hal::Lenia1Player player(4, 1.0, DeltaKernel, [](double /*potential*/)
    {
        return 0.0;
    });

    std::vector<double> pattern(static_cast<size_t>(player.Length()));
    for (int32_t x = 0; x < player.Xdim(); x++)
    {
        for (int32_t y = 0; y < player.Ydim(); y++)
        {
            double value = 0.5 + 0.4 * std::sin(static_cast<double>(x)) * std::cos(static_cast<double>(y));
            pattern[static_cast<size_t>(x * player.Ydim() + y)] = value;
            player.Set(x, y, value);
        }
    }

    player.Update();

    for (int32_t x = 0; x < player.Xdim(); x++)
    {
        for (int32_t y = 0; y < player.Ydim(); y++)
        {
            CAPTURE(x);
            CAPTURE(y);
            double expected = pattern[static_cast<size_t>(x * player.Ydim() + y)];
            CHECK(player.GetConvolvedFieldVal(x, y) == doctest::Approx(expected).epsilon(1e-9));
        }
    }
}

TEST_CASE("Hal::Lenia1Player: growth integrates into the field with clamping to [0,1] and advances the tick")
{
    // Constant positive growth of 0.2 per unit time, deltaT = 0.5, so each
    // step adds 0.1 to a constant field until it saturates at 1.
    Hal::Lenia1Player player(4, 0.5, DiscKernel, [](double /*potential*/)
    {
        return 0.2;
    });
    for (int32_t i = 0; i < player.Length(); i++)
    {
        player.Set(i, 0.5);
    }

    player.Update();
    CHECK(player.GetTick() == 1);
    CHECK(player.Get(0) == doctest::Approx(0.6).epsilon(1e-9));

    player.Update();
    CHECK(player.Get(0) == doctest::Approx(0.7).epsilon(1e-9));

    // Drive to saturation and confirm the clamp holds at 1.0.
    for (int32_t step = 0; step < 20; step++)
    {
        player.Update();
    }
    for (int32_t i = 0; i < player.Length(); i++)
    {
        CAPTURE(i);
        CHECK(player.Get(i) == doctest::Approx(1.0).epsilon(1e-12));
    }
}

TEST_CASE("Hal::Lenia1Player: negative growth clamps the field at 0")
{
    Hal::Lenia1Player player(4, 1.0, DiscKernel, [](double /*potential*/)
    {
        return -5.0;
    });
    for (int32_t i = 0; i < player.Length(); i++)
    {
        player.Set(i, 0.3);
    }

    player.Update();
    for (int32_t i = 0; i < player.Length(); i++)
    {
        CAPTURE(i);
        CHECK(player.Get(i) == doctest::Approx(0.0).epsilon(1e-12));
    }
}

TEST_CASE("Hal::Lenia1Player: GetKernelVal returns the un-normalized kernel and SetGrowthFunction swaps growth")
{
    Hal::Lenia1Player player(4, 1.0, DeltaKernel, [](double /*potential*/)
    {
        return 0.0;
    });
    // Delta kernel: only the focal cell (center) is non-zero, value 1.
    int32_t cx = player.Xdim() / 2;
    int32_t cy = player.Ydim() / 2;
    CHECK(player.GetKernelVal(cx, cy) == doctest::Approx(1.0));
    CHECK(player.GetKernelVal(0, 0) == doctest::Approx(0.0));

    for (int32_t i = 0; i < player.Length(); i++)
    {
        player.Set(i, 0.5);
    }
    player.SetGrowthFunction([](double /*potential*/)
    {
        return 0.1;
    });
    player.Update();
    CHECK(player.Get(0) == doctest::Approx(0.6).epsilon(1e-9));
}

TEST_CASE("Hal::Lenia1Player: ResetTick zeroes the tick and SetKernelFunction recomputes the kernel")
{
    Hal::Lenia1Player player(3, 0.5, DiscKernel, [](double /*potential*/)
                             { return 0.0; });
    player.Update();
    player.Update();
    CHECK(player.GetTick() == 2);
    player.ResetTick();
    CHECK(player.GetTick() == 0);

    double center = player.GetKernelVal(player.Xdim() / 2, player.Ydim() / 2);
    player.SetKernelFunction(DeltaKernel);
    CHECK(player.GetKernelVal(player.Xdim() / 2, player.Ydim() / 2) == doctest::Approx(DeltaKernel(0.0)));
    CHECK(center == doctest::Approx(DiscKernel(0.0)));
}
