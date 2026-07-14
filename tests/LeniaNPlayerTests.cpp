#include <doctest/doctest.h>

#include "Hal/Tools/LeniaNPlayer.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

// A per-pair delta kernel: 1 at the focal cell, 0 elsewhere. Once
// normalized this is a Dirac delta, so K_ij * A_j == A_j (identity).
static double DeltaKernel(int32_t /*i*/, int32_t /*j*/, double distance)
{
    return (distance < 0.5) ? 1.0 : 0.0;
}

static double DiscKernel(int32_t /*i*/, int32_t /*j*/, double distance)
{
    return (distance <= 2.0) ? 1.0 : 0.0;
}

TEST_CASE("Hal::LeniaNPlayer: dimensions, channel count, and tick initialize correctly")
{
    Hal::LeniaNPlayer player(4, 3, 0.5, DiscKernel, [](int32_t /*i*/, const std::vector<double> & /*conv*/)
    {
        return 0.0;
    });
    CHECK(player.NPlayers() == 3);
    CHECK(player.Xdim() == 16);
    CHECK(player.Ydim() == 16);
    CHECK(player.Length() == 16 * 16);
    CHECK(player.GetTick() == 0);
}

TEST_CASE("Hal::LeniaNPlayer: a per-pair delta kernel makes every convolution the identity")
{
    const int32_t nPlayers = 2;
    Hal::LeniaNPlayer player(4, nPlayers, 1.0, DeltaKernel, [](int32_t /*i*/, const std::vector<double> & /*conv*/)
    {
        return 0.0;
    });

    // Give each channel a distinct pattern.
    for (int32_t x = 0; x < player.Xdim(); x++)
    {
        for (int32_t y = 0; y < player.Ydim(); y++)
        {
            player.Set(0, x, y, 0.5 + 0.3 * std::sin(static_cast<double>(x + y)));
            player.Set(1, x, y, 0.4 + 0.2 * std::cos(static_cast<double>(x)));
        }
    }

    player.Update();

    // K_ij * A_j == A_j for every (i, j): the convolution of channel i with
    // source j reproduces source field j exactly.
    for (int32_t i = 0; i < nPlayers; i++)
    {
        for (int32_t j = 0; j < nPlayers; j++)
        {
            for (int32_t x = 0; x < player.Xdim(); x++)
            {
                for (int32_t y = 0; y < player.Ydim(); y++)
                {
                    CAPTURE(i);
                    CAPTURE(j);
                    CAPTURE(x);
                    CAPTURE(y);
                    CHECK(player.GetConvolvedFieldVal(i, j, x, y) == doctest::Approx(player.Get(j, x, y)).epsilon(1e-9));
                }
            }
        }
    }
}

TEST_CASE("Hal::LeniaNPlayer: growth couples channels — each sees all sources' convolutions")
{
    // With a delta kernel, convolution is identity, so at each cell the
    // growth for channel i receives [A_0, A_1]. A growth function that
    // returns the *other* channel's value drives cross-channel coupling we
    // can predict exactly.
    const int32_t nPlayers = 2;
    const double deltaT = 0.5;
    Hal::LeniaNPlayer player(4, nPlayers, deltaT, DeltaKernel, [](int32_t channel, const std::vector<double> &conv)
    {
        // Channel 0 grows toward channel 1's value; channel 1 toward
        // channel 0's.
        return (channel == 0) ? conv[1] : conv[0];
    });

    for (int32_t k = 0; k < player.Length(); k++)
    {
        player.Set(0, k, 0.2);
        player.Set(1, k, 0.6);
    }

    player.Update();

    // channel 0: 0.2 + conv[1](=0.6) * 0.5 = 0.5
    // channel 1: 0.6 + conv[0](=0.2) * 0.5 = 0.7
    for (int32_t k = 0; k < player.Length(); k++)
    {
        CAPTURE(k);
        CHECK(player.Get(0, k) == doctest::Approx(0.5).epsilon(1e-9));
        CHECK(player.Get(1, k) == doctest::Approx(0.7).epsilon(1e-9));
    }
    CHECK(player.GetTick() == 1);
}

TEST_CASE("Hal::LeniaNPlayer: a normalized kernel convolving a constant channel leaves it constant, and growth clamps")
{
    const int32_t nPlayers = 2;
    Hal::LeniaNPlayer player(4, nPlayers, 1.0, DiscKernel, [](int32_t /*channel*/, const std::vector<double> &conv)
    {
        // Grow by the sum of the two convolutions (drives toward clamp).
        return conv[0] + conv[1];
    });

    for (int32_t k = 0; k < player.Length(); k++)
    {
        player.Set(0, k, 0.5);
        player.Set(1, k, 0.5);
    }

    player.Update();
    // Each convolution of a constant 0.5 field with a normalized kernel is
    // 0.5, so both channels grow by 0.5+0.5 = 1.0 * dt(1.0) -> clamp at 1.
    for (int32_t k = 0; k < player.Length(); k++)
    {
        CAPTURE(k);
        CHECK(player.GetConvolvedFieldVal(0, 0, k) == doctest::Approx(0.5).epsilon(1e-9));
        CHECK(player.GetConvolvedFieldVal(1, 0, k) == doctest::Approx(0.5).epsilon(1e-9));
        CHECK(player.Get(0, k) == doctest::Approx(1.0).epsilon(1e-9));
        CHECK(player.Get(1, k) == doctest::Approx(1.0).epsilon(1e-9));
    }
}

TEST_CASE("Hal::LeniaNPlayer: GetKernelVal returns the un-normalized per-pair kernel")
{
    Hal::LeniaNPlayer player(4, 2, 1.0, DeltaKernel, [](int32_t /*channel*/, const std::vector<double> & /*conv*/)
    {
        return 0.0;
    });
    int32_t cx = player.Xdim() / 2;
    int32_t cy = player.Ydim() / 2;
    // Delta kernel: only the focal cell is non-zero, for every pair.
    for (int32_t i = 0; i < 2; i++)
    {
        for (int32_t j = 0; j < 2; j++)
        {
            CAPTURE(i);
            CAPTURE(j);
            CHECK(player.GetKernelVal(i, j, cx, cy) == doctest::Approx(1.0));
            CHECK(player.GetKernelVal(i, j, 0, 0) == doctest::Approx(0.0));
        }
    }
}

TEST_CASE("Hal::LeniaNPlayer: StoreKernel/LoadKernel restores the exact convolution state without recomputing")
{
    const int32_t nPlayers = 2;
    Hal::LeniaNPlayer player(4, nPlayers, 1.0, DeltaKernel, [](int32_t /*i*/, const std::vector<double> & /*conv*/)
    {
        return 0.0;
    });

    for (int32_t x = 0; x < player.Xdim(); x++)
    {
        for (int32_t y = 0; y < player.Ydim(); y++)
        {
            player.Set(0, x, y, 0.5 + 0.3 * std::sin(static_cast<double>(x + y)));
            player.Set(1, x, y, 0.4 + 0.2 * std::cos(static_cast<double>(x)));
        }
    }

    int32_t cx = player.Xdim() / 2;
    int32_t cy = player.Ydim() / 2;
    player.StoreKernel("delta");

    // Switching kernels changes the stored kernel weights (the cell next
    // to the focal center is inside the disc but outside the delta)...
    player.SetKernelFunction(DiscKernel);
    CHECK(player.GetKernelVal(0, 0, cx + 1, cy) == doctest::Approx(1.0));
    player.StoreKernel("disc");

    // ...and loading the snapshot restores the delta kernel exactly: the
    // neighbor weight is back to 0 and one Update convolves as the
    // identity again (the delta-kernel ground truth from the case above).
    player.LoadKernel("delta");
    CHECK(player.GetKernelVal(0, 0, cx + 1, cy) == doctest::Approx(0.0));
    player.Update();
    for (int32_t i = 0; i < nPlayers; i++)
    {
        for (int32_t j = 0; j < nPlayers; j++)
        {
            for (int32_t x = 0; x < player.Xdim(); x++)
            {
                for (int32_t y = 0; y < player.Ydim(); y++)
                {
                    CHECK(player.GetConvolvedFieldVal(i, j, x, y) == doctest::Approx(player.GetField(j).Get(x, y)).epsilon(1e-9));
                }
            }
        }
    }

    // Snapshots survive multiple loads and switching back and forth.
    player.LoadKernel("disc");
    CHECK(player.GetKernelVal(0, 0, cx + 1, cy) == doctest::Approx(1.0));
    player.LoadKernel("delta");
    CHECK(player.GetKernelVal(0, 0, cx + 1, cy) == doctest::Approx(0.0));

    // Unknown labels throw rather than silently doing nothing.
    CHECK_THROWS_AS(player.LoadKernel("never-stored"), std::invalid_argument);
}

TEST_CASE("Hal::FftGrid::Clone is a deep copy")
{
    Hal::Grid2D<double> source(8, 8);
    for (int32_t x = 0; x < 8; x++)
    {
        for (int32_t y = 0; y < 8; y++)
        {
            source.Set(x, y, 0.1 * x - 0.3 * y);
        }
    }
    Hal::FftGrid original(source);
    Hal::FftGrid copy = original.Clone();

    for (int32_t x = 0; x < 8; x++)
    {
        for (int32_t y = 0; y < 8; y++)
        {
            CHECK(copy.GetReal(x, y) == original.GetReal(x, y));
            CHECK(copy.GetImag(x, y) == original.GetImag(x, y));
        }
    }

    // Transforming the copy leaves the original untouched.
    copy.Fft2();
    CHECK(original.GetReal(1, 2) == doctest::Approx(source.Get(1, 2)));
}
