#include <doctest/doctest.h>

#include "Hal/Grids/GridGeometry3D.h"
#include "Hal/Neighborhood.h"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------
// Flood-fill mask helpers ported from Interfaces/Grid3D.java. The geometry
// makes the expected result countable by hand: a solid 3x3x3 masked cube
// sits in the interior of a 5x5x5 grid, so under von-Neumann connectivity
// its 27 cells are all mutually reachable, and its outward-facing shell is
// exactly the 26 surface cells (every cube cell except the fully-enclosed
// center).
//
// Note the faithful Java quirk (JAVA_BUGS_FOUND.md #17): BucketFillMask
// never marks the seed cell itself in `out`, so filling from the cube's
// center yields the other 26 cube cells, not all 27.
// ---------------------------------------------------------------------

// True for the 3x3x3 cube spanning [1,3] on each axis.
static bool InCube(int32_t x,
                   int32_t y,
                   int32_t z)
{
    return (x >= 1) && (x <= 3) && (y >= 1) && (y <= 3) && (z >= 1) && (z <= 3);
}

TEST_CASE("GridGeometry3D::BucketFillMask fills exactly the connected masked region")
{
    Hal::GridGeometry3D geom(5, 5, 5, false, false, false);
    std::vector<int32_t> hood = Hal::VonNeumannHood3D(false);

    // std::vector<bool> is a bitset (no addressable bool*), so use char buffers.
    std::vector<char> out(geom.Length(), 0);
    std::vector<char> scratch(geom.Length(), 0);
    std::vector<int32_t> stack;

    int32_t center = geom.I(2, 2, 2);
    geom.BucketFillMask(reinterpret_cast<bool *>(out.data()),
                        hood.data(),
                        static_cast<int32_t>(hood.size()),
                        reinterpret_cast<bool *>(scratch.data()),
                        stack,
                        center,
                        [](int32_t, int32_t x, int32_t y, int32_t z) { return InCube(x, y, z); });

    int32_t filled = 0;
    for (int32_t x = 0; x < 5; x++)
    {
        for (int32_t y = 0; y < 5; y++)
        {
            for (int32_t z = 0; z < 5; z++)
            {
                bool isSet = out[geom.I(x, y, z)] != 0;
                // Masked everywhere in the cube except the unmarked seed.
                bool expected = InCube(x, y, z) && !((x == 2) && (y == 2) && (z == 2));
                CAPTURE(x);
                CAPTURE(y);
                CAPTURE(z);
                CHECK(isSet == expected);
                if (isSet)
                {
                    filled++;
                }
            }
        }
    }
    // 27 cube cells minus the seed the algorithm never marks.
    CHECK(filled == 26);
    CHECK(out[center] == 0);
}

TEST_CASE("GridGeometry3D::BucketFillMask does not reach a disconnected masked cell")
{
    Hal::GridGeometry3D geom(5, 5, 5, false, false, false);
    std::vector<int32_t> hood = Hal::VonNeumannHood3D(false);

    std::vector<char> out(geom.Length(), 0);
    std::vector<char> scratch(geom.Length(), 0);
    std::vector<int32_t> stack;

    // Mask = the cube plus a lone cell in the far corner, unreachable from
    // the cube under von-Neumann connectivity.
    int32_t loner = geom.I(0, 0, 4);
    auto mask = [&](int32_t i, int32_t x, int32_t y, int32_t z) { return InCube(x, y, z) || (i == loner); };

    geom.BucketFillMask(reinterpret_cast<bool *>(out.data()),
                        hood.data(),
                        static_cast<int32_t>(hood.size()),
                        reinterpret_cast<bool *>(scratch.data()),
                        stack,
                        geom.I(2, 2, 2),
                        mask);

    CHECK(out[loner] == 0);
    // The seed (2,2,2) is never marked, but its masked neighbors are.
    CHECK(out[geom.I(2, 2, 2)] == 0);
    CHECK(out[geom.I(1, 2, 2)] != 0);
}

TEST_CASE("GridGeometry3D::BorderFindMask marks the outward-facing shell of the masked region")
{
    Hal::GridGeometry3D geom(5, 5, 5, false, false, false);
    std::vector<int32_t> hood = Hal::VonNeumannHood3D(false);

    std::vector<char> out(geom.Length(), 0);
    std::vector<char> scratch(geom.Length(), 0);
    std::vector<int32_t> stack;

    // Start in the open region outside the cube.
    geom.BorderFindMask(reinterpret_cast<bool *>(out.data()),
                        hood.data(),
                        static_cast<int32_t>(hood.size()),
                        reinterpret_cast<bool *>(scratch.data()),
                        stack,
                        geom.I(0, 0, 0),
                        [](int32_t, int32_t x, int32_t y, int32_t z) { return InCube(x, y, z); });

    int32_t marked = 0;
    for (int32_t i = 0; i < geom.Length(); i++)
    {
        if (out[i] != 0)
        {
            marked++;
        }
    }
    // Every cube cell except the fully-enclosed center (2,2,2).
    CHECK(marked == 26);
    CHECK(out[geom.I(2, 2, 2)] == 0);
    CHECK(out[geom.I(1, 2, 2)] != 0);
}
