#include <doctest/doctest.h>

#include "Hal/Grids/Grid1D.h"
#include "Hal/Grids/Grid2D.h"
#include "Hal/Grids/Grid3D.h"
#include "Hal/Grids/GridGeometry2D.h"
#include "Hal/Grids/GridGeometry3D.h"
#include "Hal/Neighborhood.h"

#include <vector>

TEST_CASE("Grid2D Get/Set/Add operate correctly by index and by coordinate")
{
    Hal::Grid2D<double> grid(4, 5);
    CHECK(grid.Xdim() == 4);
    CHECK(grid.Ydim() == 5);
    CHECK(grid.Length() == 20);

    grid.Set(2, 3, 7.5);
    CHECK(grid.Get(2, 3) == doctest::Approx(7.5));
    CHECK(grid.Get(grid.I(2, 3)) == doctest::Approx(7.5));

    grid.Add(2, 3, 2.5);
    CHECK(grid.Get(2, 3) == doctest::Approx(10.0));

    grid.Scale(2, 3, 2.0);
    CHECK(grid.Get(2, 3) == doctest::Approx(20.0));
}

TEST_CASE("Grid2D ItoX/ItoY round-trip with I()")
{
    Hal::Grid2D<int32_t> grid(6, 7);
    for (int32_t x = 0; x < 6; x++)
    {
        for (int32_t y = 0; y < 7; y++)
        {
            int32_t i = grid.I(x, y);
            CHECK(grid.ItoX(i) == x);
            CHECK(grid.ItoY(i) == y);
        }
    }
}

TEST_CASE("Grid2D SetAll/AddAll/MulAll/BoundAll/GetSum/GetAvg/GetMax/GetMin")
{
    Hal::Grid2D<double> grid(2, 2);
    grid.SetAll(3.0);
    CHECK(grid.GetSum() == doctest::Approx(12.0));
    CHECK(grid.GetAvg() == doctest::Approx(3.0));

    grid.AddAll(1.0);
    CHECK(grid.Get(0) == doctest::Approx(4.0));

    grid.MulAll(2.0);
    CHECK(grid.Get(0) == doctest::Approx(8.0));

    grid.Set(0, 100.0);
    grid.BoundAll(0.0, 10.0);
    CHECK(grid.Get(0) == doctest::Approx(10.0));
    CHECK(grid.GetMax() == doctest::Approx(10.0));
    CHECK(grid.GetMin() == doctest::Approx(8.0));
}

TEST_CASE("Grid2D<int32_t> GetAvg truncates like HAL's Java Grid2Dint")
{
    Hal::Grid2D<int32_t> grid(1, 3);
    grid.Set(0, 1);
    grid.Set(1, 1);
    grid.Set(2, 2);
    // sum=4, length=3 -> integer division truncates to 1.
    CHECK(grid.GetAvg() == 1);
}

TEST_CASE("Grid2D wraparound: In/InWrap/Wrap behave correctly at the boundary")
{
    Hal::Grid2D<double> grid(4, 4, true, false);
    CHECK_FALSE(grid.In(4, 0));
    CHECK(grid.InWrap(4, 0));
    CHECK(grid.WrapX(4) == 0);
    CHECK(grid.WrapX(-1) == 3);
}

TEST_CASE("Grid2D MapHood maps a neighborhood template onto a concrete center, dropping out-of-bounds entries without wrap")
{
    Hal::Grid2D<double> grid(3, 3);
    std::vector<int32_t> hood = Hal::MooreHood(false);

    // Center square (1,1) is fully interior: all 8 Moore neighbors are valid.
    int32_t count = grid.MapHood(hood.data(), static_cast<int32_t>(hood.size()), grid.I(1, 1));
    CHECK(count == 8);

    // Corner square (0,0) without wraparound only has 3 valid Moore neighbors.
    std::vector<int32_t> cornerHood = Hal::MooreHood(false);
    int32_t cornerCount = grid.MapHood(cornerHood.data(), static_cast<int32_t>(cornerHood.size()), grid.I(0, 0));
    CHECK(cornerCount == 3);
}

TEST_CASE("Grid2D ApplyHood visits exactly the mapped, in-bounds neighbors")
{
    Hal::Grid2D<double> grid(3, 3);
    std::vector<int32_t> hood = Hal::VonNeumannHood(false);

    int32_t visited = 0;
    grid.ApplyHood(hood.data(), static_cast<int32_t>(hood.size()), 1, 1, [&](int32_t x, int32_t y)
                   {
        visited++;
        CHECK(grid.In(x, y)); });
    CHECK(visited == 4);
}

TEST_CASE("Grid2D BoundaryIs returns one index per boundary square")
{
    Hal::Grid2D<double> grid(3, 4);
    std::vector<int32_t> boundary = grid.BoundaryIs();
    CHECK(boundary.size() == static_cast<size_t>((3 + 4) * 2));
}

TEST_CASE("Grid1D basic indexing and wraparound")
{
    Hal::Grid1D<double> grid(5, true);
    grid.Set(2, 9.0);
    CHECK(grid.Get(2) == doctest::Approx(9.0));
    CHECK(grid.WrapX(5) == 0);
    CHECK(grid.WrapX(-1) == 4);
    CHECK(grid.InWrap(10));
}

TEST_CASE("Grid3D basic indexing round-trips through ItoX/ItoY/ItoZ")
{
    Hal::Grid3D<double> grid(3, 4, 5);
    CHECK(grid.Length() == 60);
    for (int32_t x = 0; x < 3; x++)
    {
        for (int32_t y = 0; y < 4; y++)
        {
            for (int32_t z = 0; z < 5; z++)
            {
                int32_t i = grid.I(x, y, z);
                CHECK(grid.ItoX(i) == x);
                CHECK(grid.ItoY(i) == y);
                CHECK(grid.ItoZ(i) == z);
            }
        }
    }
}

TEST_CASE("Grid3D Get/Set/Add by coordinate")
{
    Hal::Grid3D<int32_t> grid(2, 2, 2);
    grid.Set(1, 0, 1, 5);
    CHECK(grid.Get(1, 0, 1) == 5);
    grid.Add(1, 0, 1, 3);
    CHECK(grid.Get(1, 0, 1) == 8);
}

TEST_CASE("Grid3D MapHood maps the Moore3D template onto an interior center")
{
    Hal::Grid3D<double> grid(3, 3, 3);
    std::vector<int32_t> hood = Hal::MooreHood3D(false);
    int32_t count = grid.MapHood(hood.data(), static_cast<int32_t>(hood.size()), grid.I(1, 1, 1));
    CHECK(count == 26);
}

// Regression test for a port-introduced (not Java) sign-flip bug: an
// earlier version of GridGeometry2D/3D's wrap-mode DispX/DispY/DispZ
// called DispWrap with its arguments swapped relative to the non-wrap
// branch, so wrapping around the boundary flipped the displacement's
// sign instead of just picking the shorter path. Checked here directly
// rather than through a squared-distance query, since squared distance
// can't detect a sign flip.
TEST_CASE("GridGeometry2D::DispX/DispY agree in sign with the non-wrap case when a wrap shortcut is taken")
{
    Hal::GridGeometry2D grid(10, 10, true, true);

    CHECK(grid.DispX(1.0, 9.0) == doctest::Approx(-2.0));
    CHECK(grid.DispX(9.0, 1.0) == doctest::Approx(2.0));
    CHECK(grid.DispY(1.0, 9.0) == doctest::Approx(-2.0));
    CHECK(grid.DispY(9.0, 1.0) == doctest::Approx(2.0));

    CHECK(grid.DispX(2.0, 5.0) == doctest::Approx(3.0));
    CHECK(grid.DispX(5.0, 2.0) == doctest::Approx(-3.0));
}

TEST_CASE("GridGeometry3D::DispX/DispY/DispZ agree in sign with the non-wrap case when a wrap shortcut is taken")
{
    Hal::GridGeometry3D grid(10, 10, 10, true, true, true);

    CHECK(grid.DispX(1.0, 9.0) == doctest::Approx(-2.0));
    CHECK(grid.DispX(9.0, 1.0) == doctest::Approx(2.0));
    CHECK(grid.DispY(1.0, 9.0) == doctest::Approx(-2.0));
    CHECK(grid.DispY(9.0, 1.0) == doctest::Approx(2.0));
    CHECK(grid.DispZ(1.0, 9.0) == doctest::Approx(-2.0));
    CHECK(grid.DispZ(9.0, 1.0) == doctest::Approx(2.0));
}
