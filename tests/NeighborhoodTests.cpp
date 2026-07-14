#include <doctest/doctest.h>

#include "Hal/Neighborhood.h"

#include <utility>
#include <vector>

// Each hood table is [padding zeros, one per neighbor] followed by the
// (x,y[,z]) offset pairs/triples themselves, so the total length is always
// paddingCount + paddingCount * dims. These tests pin down that shape plus
// a couple of known offset values, so a future refactor of MapHood's
// consumers can't silently break the padding convention Phase 2 depends on.

TEST_CASE("VonNeumannHood has 4 neighbors (8 without origin, 5 with)")
{
    std::vector<int32_t> withoutOrigin = Hal::VonNeumannHood(false);
    REQUIRE(withoutOrigin.size() == 4 + 4 * 2);
    // Padding is the first 4 entries; offsets start at index 4.
    CHECK(withoutOrigin[4] == 1);
    CHECK(withoutOrigin[5] == 0);

    std::vector<int32_t> withOrigin = Hal::VonNeumannHood(true);
    REQUIRE(withOrigin.size() == 5 + 5 * 2);
    CHECK(withOrigin[5] == 0);
    CHECK(withOrigin[6] == 0);
}

TEST_CASE("MooreHood has 8 neighbors (16 without origin, 18 with)")
{
    std::vector<int32_t> withoutOrigin = Hal::MooreHood(false);
    REQUIRE(withoutOrigin.size() == 8 + 8 * 2);

    std::vector<int32_t> withOrigin = Hal::MooreHood(true);
    REQUIRE(withOrigin.size() == 9 + 9 * 2);
}

TEST_CASE("MooreHood3D has 26 neighbors (52 without origin, 54 with)")
{
    std::vector<int32_t> withoutOrigin = Hal::MooreHood3D(false);
    REQUIRE(withoutOrigin.size() == 26 + 26 * 3);

    std::vector<int32_t> withOrigin = Hal::MooreHood3D(true);
    REQUIRE(withOrigin.size() == 27 + 27 * 3);
}

TEST_CASE("VonNeumannHood3D has 6 neighbors")
{
    std::vector<int32_t> withoutOrigin = Hal::VonNeumannHood3D(false);
    REQUIRE(withoutOrigin.size() == 6 + 6 * 3);
}

TEST_CASE("DiagonalHood3D has 8 neighbors, all offsets +/-1 in every axis")
{
    std::vector<int32_t> withoutOrigin = Hal::DiagonalHood3D(false);
    REQUIRE(withoutOrigin.size() == 8 + 8 * 3);
    for (size_t i = 8; i < withoutOrigin.size(); i++)
    {
        CHECK((withoutOrigin[i] == 1 || withoutOrigin[i] == -1));
    }
}

TEST_CASE("CircleHood(false, 1.0) is the 4 von-Neumann offsets in padded layout")
{
    std::vector<int32_t> hood = Hal::CircleHood(false, 1.0);
    // 4 neighbors: ct leading padding ints then ct (x,y) pairs.
    REQUIRE(hood.size() == 4 + 4 * 2);
    int32_t ct = 4;
    std::vector<std::pair<int32_t, int32_t>> got;
    for (int32_t k = 0; k < ct; k++)
    {
        got.emplace_back(hood[ct + k * 2], hood[ct + k * 2 + 1]);
    }
    auto has = [&](int32_t x, int32_t y) {
        for (const auto &p : got)
        {
            if ((p.first == x) && (p.second == y))
            {
                return true;
            }
        }
        return false;
    };
    CHECK(has(1, 0));
    CHECK(has(-1, 0));
    CHECK(has(0, 1));
    CHECK(has(0, -1));
    CHECK_FALSE(has(1, 1));
}

TEST_CASE("CircleHood only includes offsets within the radius, origin optional and first")
{
    std::vector<int32_t> withOrigin = Hal::CircleHood(true, 2.0);
    int32_t ct = static_cast<int32_t>(withOrigin.size()) / 3;
    // Origin is listed first among the pairs.
    CHECK(withOrigin[ct + 0] == 0);
    CHECK(withOrigin[ct + 1] == 0);
    // Every offset satisfies x^2 + y^2 <= r^2.
    for (int32_t k = 0; k < ct; k++)
    {
        int32_t x = withOrigin[ct + k * 2];
        int32_t y = withOrigin[ct + k * 2 + 1];
        CHECK((x * x + y * y) <= 4);
    }
    // Without the origin it has exactly one fewer entry.
    std::vector<int32_t> noOrigin = Hal::CircleHood(false, 2.0);
    CHECK(static_cast<int32_t>(noOrigin.size()) / 3 == ct - 1);
}

TEST_CASE("RectangleHood covers the full axis-aligned rectangle in padded layout")
{
    // 3x5 rectangle (radX=1, radY=2) without origin: 3*5 - 1 = 14 offsets.
    std::vector<int32_t> hood = Hal::RectangleHood(false, 1, 2);
    int32_t ct = static_cast<int32_t>(hood.size()) / 3;
    CHECK(ct == 14);
    for (int32_t k = 0; k < ct; k++)
    {
        int32_t x = hood[ct + k * 2];
        int32_t y = hood[ct + k * 2 + 1];
        CHECK(x >= -1);
        CHECK(x <= 1);
        CHECK(y >= -2);
        CHECK(y <= 2);
        bool isOrigin = (x == 0) && (y == 0);
        CHECK_FALSE(isOrigin);
    }
    // With origin it has exactly one more entry, listed first.
    std::vector<int32_t> withOrigin = Hal::RectangleHood(true, 1, 2);
    int32_t ctO = static_cast<int32_t>(withOrigin.size()) / 3;
    CHECK(ctO == 15);
    CHECK(withOrigin[ctO + 0] == 0);
    CHECK(withOrigin[ctO + 1] == 0);
}
