#include <doctest/doctest.h>

#include "Hal/Grids/Grid1D.h"
#include "Hal/Grids/Grid2D.h"
#include "Hal/Grids/Grid3D.h"
#include "Hal/Grids/GridGeometry1D.h"
#include "Hal/Grids/GridGeometry2D.h"
#include "Hal/Grids/GridGeometry3D.h"
#include "Hal/Neighborhood.h"

#include "Hal/Rand.h"

#include <set>
#include <stdexcept>
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

TEST_CASE("Grid2D<int32_t> GetAvg truncates")
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

// Wrapped displacements must keep the non-wrap sign convention; checked
// directly because squared distances can't see a sign flip.
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

TEST_CASE("GridGeometry1D::DispX agrees in sign with the non-wrap case when a wrap shortcut is taken")
{
    Hal::GridGeometry1D grid(10, true);

    CHECK(grid.DispX(1.0, 9.0) == doctest::Approx(-2.0));
    CHECK(grid.DispX(9.0, 1.0) == doctest::Approx(2.0));
    CHECK(grid.DispX(2.0, 5.0) == doctest::Approx(3.0));
    CHECK(grid.DispX(5.0, 2.0) == doctest::Approx(-3.0));
}

TEST_CASE("GridGeometry1D: WrapI, hood mapping, and neighborhood helpers")
{
    Hal::GridGeometry1D wrapped(5, true);
    Hal::GridGeometry1D bounded(5, false);
    CHECK(wrapped.WrapI(-1) == 4);
    CHECK(wrapped.WrapI(7) == 2);
    CHECK(bounded.WrapI(3) == 3);
    CHECK_THROWS_AS(bounded.WrapI(5), std::invalid_argument);

    std::vector<int32_t> hood = {0, 0, -1, 1};
    CHECK(bounded.GetHoodI(hood.data(), 4, 0, 0) == -1);
    CHECK(wrapped.GetHoodI(hood.data(), 4, 0, 0) == 4);
    CHECK(bounded.ContainsValidI(hood.data(), 4, 0, [](int32_t x) { return x == 1; }));
    CHECK_FALSE(bounded.ContainsValidI(hood.data(), 4, 0, [](int32_t x) { return x == 4; }));

    std::vector<int32_t> visited;
    wrapped.ApplyHoodWithIndex(hood.data(), 4, 0, [&](int32_t entry, int32_t x) { visited.push_back(entry * 10 + x); });
    CHECK(visited == std::vector<int32_t>{4, 11});

    std::vector<int32_t> evalHood = hood;
    int32_t count = bounded.MapHood(evalHood.data(), 4, 2, [](int32_t x) { return x > 2; });
    REQUIRE(count == 1);
    CHECK(evalHood[0] == 3);

    std::vector<int32_t> rect;
    wrapped.ApplyRectangle(3, 4, [&](int32_t x) { rect.push_back(x); });
    CHECK(rect == std::vector<int32_t>{3, 4, 0, 1});

    std::vector<int32_t> edges;
    bounded.ForEdge([&](int32_t x) { edges.push_back(x); });
    CHECK(edges == std::vector<int32_t>{0, 4});

    Hal::GridGeometry1D coarse(10, false);
    CHECK(bounded.MapXsq(4, coarse) == 9);
    CHECK(bounded.MapXpt(2.5, coarse) == doctest::Approx(5.0));

    Hal::Rand rand(1);
    for (int32_t i = 0; i < 20; i++)
    {
        int32_t x = wrapped.RandomHoodI(hood.data(), 4, 0, rand);
        CHECK(((x == 4) || (x == 1)));
    }
}

TEST_CASE("GridGeometry2D: WrapI, Map*, GetHoodI, ContainsValidI, and ApplyRectangle")
{
    Hal::GridGeometry2D grid(4, 3, true, false);
    CHECK(grid.WrapI(-1, 1) == grid.I(3, 1));
    CHECK_THROWS_AS(grid.WrapI(0, 3), std::invalid_argument);

    Hal::GridGeometry2D fine(8, 6, false, false);
    CHECK(grid.MapXsq(1, fine) == 3);
    CHECK(grid.MapYsq(2, fine) == 5);
    CHECK(grid.MapI(grid.I(1, 2), fine) == fine.I(3, 5));
    CHECK(grid.MapXpt(1.5, fine) == doctest::Approx(3.0));
    CHECK(grid.MapYpt(1.5, fine) == doctest::Approx(3.0));

    std::vector<int32_t> hood = Hal::VonNeumannHood(false);
    int32_t length = static_cast<int32_t>(hood.size());
    CHECK(grid.GetHoodI(hood.data(), length, 3, 0, 0) == grid.I(0, 0));
    CHECK(grid.GetHoodI(hood.data(), length, 0, 0, 3) == -1);
    CHECK(grid.ContainsValidI(hood.data(), length, 0, 0, [](int32_t x, int32_t y) { return (x == 3) && (y == 0); }));

    std::set<int32_t> rect;
    grid.ApplyRectangle(3, 1, 2, 2, [&](int32_t x, int32_t y) { rect.insert(grid.I(x, y)); });
    CHECK(rect == std::set<int32_t>{grid.I(3, 1), grid.I(3, 2), grid.I(0, 1), grid.I(0, 2)});
}

TEST_CASE("GridGeometry2D: ForEdge visits each boundary square once and BoundaryIs lists rows then columns")
{
    Hal::GridGeometry2D grid(5, 4, false, false);
    std::set<int32_t> visited;
    int32_t calls = 0;
    grid.ForEdge([&](int32_t i, int32_t x, int32_t y) {
        CHECK(i == grid.I(x, y));
        visited.insert(i);
        calls++;
    });
    CHECK(calls == (2 * 5 + 2 * 4 - 4));
    CHECK(visited.size() == static_cast<size_t>(calls));

    std::vector<int32_t> boundary = grid.BoundaryIs();
    REQUIRE(boundary.size() == static_cast<size_t>((5 + 4) * 2));
    CHECK(boundary[0] == grid.I(0, 0));
    CHECK(boundary[5] == grid.I(0, 3));
    CHECK(boundary[10] == grid.I(0, 0));
    CHECK(boundary[14] == grid.I(4, 0));
}

TEST_CASE("GridGeometry2D::AlongLineIs lists every square a segment crosses")
{
    Hal::GridGeometry2D grid(10, 10, false, false);
    std::vector<int32_t> squares(32);
    int32_t count = grid.AlongLineIs(0.5, 2.5, 4.5, 2.5, squares.data());
    REQUIRE(count == 5);
    for (int32_t i = 0; i < count; i++)
    {
        CHECK(squares[static_cast<size_t>(i)] == grid.I(i, 2));
    }

    count = grid.AlongLineIs(0.5, 0.5, 2.5, 2.5, squares.data());
    CHECK(squares[0] == grid.I(0, 0));
    CHECK(squares[static_cast<size_t>(count - 1)] == grid.I(2, 2));
    CHECK(count == 5);
}

TEST_CASE("GridGeometry3D: WrapI, ConvI, GetHoodI, ForEdge, and BoundaryIs")
{
    Hal::GridGeometry3D grid(4, 3, 2, false, false, true);
    CHECK(grid.WrapI(1, 1, -1) == grid.I(1, 1, 1));
    CHECK_THROWS_AS(grid.WrapI(4, 0, 0), std::invalid_argument);

    Hal::GridGeometry3D fine(8, 6, 4, false, false, false);
    CHECK(grid.ConvI(grid.I(1, 2, 1), fine) == fine.I(3, 5, 3));
    CHECK(grid.ConvZpt(1.0, fine) == doctest::Approx(2.0));

    std::vector<int32_t> hood = Hal::VonNeumannHood3D(false);
    int32_t length = static_cast<int32_t>(hood.size());
    CHECK(grid.GetHoodI(hood.data(), length, 1, 1, 1, 4) == grid.I(1, 1, 0));
    CHECK(grid.GetHoodI(hood.data(), length, 0, 1, 1, 1) == -1);

    std::set<int32_t> visited;
    int32_t calls = 0;
    Hal::GridGeometry3D box(4, 4, 4, false, false, false);
    box.ForEdge([&](int32_t i, int32_t, int32_t, int32_t) {
        visited.insert(i);
        calls++;
    });
    CHECK(calls == (64 - 8));
    CHECK(visited.size() == static_cast<size_t>(calls));
    CHECK(box.BoundaryIs().size() == static_cast<size_t>(6 * 16));
}

TEST_CASE("Grid1D and Grid3D aggregate helpers")
{
    Hal::Grid1D<int32_t> line(4);
    line.SetAll(2);
    line.Add(1, 4);
    line.MulAll(1.5);
    CHECK(line.GetSum() == (3 + 9 + 3 + 3));
    CHECK(line.GetMax() == 9);
    CHECK(line.GetMin() == 3);
    CHECK(line.GetAvg() == 4);
    line.BoundAll(0, 5);
    CHECK(line.GetMax() == 5);

    Hal::Grid3D<double> volume(2, 2, 2);
    volume.SetAll(1.0);
    volume.Set(1, 1, 1, 9.0);
    CHECK(volume.GetAvg() == doctest::Approx(2.0));
    CHECK(volume.Get(volume.I(1, 1, 1)) == doctest::Approx(9.0));
}
