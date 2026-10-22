#include <doctest/doctest.h>

#include "Hal/Grids/PopulationGrid1D.h"
#include "Hal/Grids/PopulationGrid2D.h"
#include "Hal/Grids/PopulationGrid3D.h"
#include "Hal/Rand.h"
#include "Hal/Tools/MultinomialCalc.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

TEST_CASE("PopulationGrid2D: Add/Update/Get follow the double-buffered delta convention")
{
    Hal::PopulationGrid2D<int32_t> grid(5, 5);
    grid.Add(2, 2, 10);
    // Not visible via Get() until Update() commits the pending delta.
    CHECK(grid.Get(2, 2) == 0);

    grid.Update();
    CHECK(grid.Get(2, 2) == 10);
    CHECK(grid.Pop() == 10);
    CHECK(grid.OccupiedArea() == 1);
}

TEST_CASE("PopulationGrid2D: Set overwrites and Move transfers population between cells")
{
    Hal::PopulationGrid2D<int32_t> grid(5, 5);
    grid.Set(1, 1, 20);
    grid.Update();
    CHECK(grid.Get(1, 1) == 20);

    grid.Move(1, 1, 3, 3, 15);
    grid.Update();
    CHECK(grid.Get(1, 1) == 5);
    CHECK(grid.Get(3, 3) == 15);
    CHECK(grid.Pop() == 20);
    CHECK(grid.OccupiedArea() == 2);
}

TEST_CASE("PopulationGrid2D: Update throws if a cell's population goes negative")
{
    Hal::PopulationGrid2D<int32_t> grid(3, 3);
    grid.Add(0, 0, 5);
    grid.Update();
    grid.Add(0, 0, -10);
    CHECK_THROWS_AS(grid.Update(), std::runtime_error);
}

TEST_CASE("PopulationGrid2D: ApplyOccupied visits exactly the occupied cells and their counts")
{
    Hal::PopulationGrid2D<int32_t> grid(4, 4);
    grid.Set(0, 0, 3);
    grid.Set(2, 3, 7);
    grid.Update();

    int64_t total = 0;
    int32_t visited = 0;
    grid.ApplyOccupied([&](int32_t i, int32_t count)
                       {
        visited++;
        total += count;
        CHECK(grid.Get(i) == count); });
    CHECK(visited == 2);
    CHECK(total == 10);
}

TEST_CASE("PopulationGrid2D: range-for iteration matches ApplyOccupied")
{
    Hal::PopulationGrid2D<int32_t> grid(4, 4);
    grid.Set(0, 0, 3);
    grid.Set(2, 3, 7);
    grid.Update();

    int64_t total = 0;
    int32_t visited = 0;
    for (int32_t i : grid)
    {
        visited++;
        total += grid.Get(i);
    }
    CHECK(visited == 2);
    CHECK(total == 10);
}

TEST_CASE("PopulationGrid2D: switches to sparse indexing at low occupancy and back to dense at higher occupancy")
{
    // length = 2000; sparse activates at occupiedArea <= 0.1% (2 cells),
    // deactivates back to dense at >= 1% (20 cells).
    Hal::PopulationGrid2D<int32_t> grid(50, 40);

    grid.Set(0, 0, 5);
    grid.Update();
    CHECK(grid.OccupiedArea() == 1);

    for (int32_t i = 1; i < 26; i++)
    {
        grid.Set(i, 0, i);
        grid.Update();
    }
    CHECK(grid.OccupiedArea() == 26);

    int64_t total = 0;
    int32_t visited = 0;
    grid.ApplyOccupied([&](int32_t, int32_t count)
                       {
        visited++;
        total += count; });
    CHECK(visited == 26);
    CHECK(total == grid.Pop());

    // Drop back down under the sparse threshold by clearing most cells.
    for (int32_t i = 1; i < 26; i++)
    {
        grid.Set(i, 0, 0);
    }
    grid.Update();
    CHECK(grid.OccupiedArea() == 1);
    CHECK(grid.Get(0, 0) == 5);
}

TEST_CASE("PopulationGrid1D: index-based API works directly (x is already the flat index)")
{
    Hal::PopulationGrid1D<int32_t> grid(10);
    grid.Set(4, 42);
    grid.Update();
    CHECK(grid.Get(4) == 42);
    CHECK(grid.Pop() == 42);
}

TEST_CASE("PopulationGrid3D: coordinate-based Get/Set/Add/Move")
{
    Hal::PopulationGrid3D<int32_t> grid(4, 4, 4);
    grid.Set(1, 2, 3, 9);
    grid.Update();
    CHECK(grid.Get(1, 2, 3) == 9);

    grid.Move(1, 2, 3, 0, 0, 0, 4);
    grid.Update();
    CHECK(grid.Get(1, 2, 3) == 5);
    CHECK(grid.Get(0, 0, 0) == 4);
}

TEST_CASE("PopulationGrid2D<int64_t>: large counts work via the templated count type")
{
    Hal::PopulationGrid2D<int64_t> grid(3, 3);
    int64_t big = static_cast<int64_t>(3) * 1000000000LL;
    grid.Set(1, 1, big);
    grid.Update();
    CHECK(grid.Get(1, 1) == big);
    CHECK(grid.Pop() == big);
}

TEST_CASE("PopulationGrid2D::Diffusion conserves total population and spreads a point mass")
{
    Hal::Rand                      rng(42);
    Hal::MultinomialCalc           mn(rng);
    Hal::PopulationGrid2D<int32_t> cells(21, 21, false, false);

    const int32_t startPop = 1000000;
    cells.Set(10, 10, startPop);
    cells.Update();

    // A few diffusion steps.
    for (int32_t step = 0; step < 10; step++)
    {
        cells.Diffusion(0.2, mn);
        cells.Update();
    }

    // Total population is conserved (mean-field diffusion moves, never
    // creates or destroys, agents; zero-flux boundaries keep them in-domain).
    int64_t total = 0;
    for (int32_t i = 0; i < cells.Xdim() * cells.Ydim(); i++)
    {
        total += cells.Get(i);
    }
    CHECK(total == startPop);

    // The center no longer holds everyone, and neighbors have received mass.
    CHECK(cells.Get(10, 10) < startPop);
    int64_t neighborSum = cells.Get(9, 10) + cells.Get(11, 10) + cells.Get(10, 9) + cells.Get(10, 11);
    CHECK(neighborSum > 0);
}

TEST_CASE("PopulationGrid2D<int64_t>::Diffusion moves counts beyond the int32 range without truncation")
{
    Hal::Rand                      rng(9);
    Hal::MultinomialCalc           mn(rng);
    Hal::PopulationGrid2D<int64_t> cells(5, 5, true, true);

    const int64_t startPop = 6000000000LL;
    cells.Set(2, 2, startPop);
    cells.Update();
    cells.Diffusion(0.1, mn);
    cells.Update();

    CHECK(cells.Pop() == startPop);
    int64_t total = 0;
    for (int32_t i = 0; i < cells.Length(); i++)
    {
        total += cells.Get(i);
    }
    CHECK(total == startPop);
    double moved = static_cast<double>(cells.Get(3, 2));
    CHECK(moved == doctest::Approx(startPop * 0.1).epsilon(0.01));
}

TEST_CASE("PopulationGridBase::SetupSparseIndices switches to sparse iteration and keeps pending deltas")
{
    Hal::PopulationGrid2D<int32_t> grid(40, 40);
    grid.Set(3, 4, 5);
    grid.Update();
    grid.Add(7, 7, 2);
    grid.Add(3, 4, 1);

    grid.SetupSparseIndices();
    grid.Update();

    CHECK(grid.Get(7, 7) == 2);
    CHECK(grid.Get(3, 4) == 6);
    CHECK(grid.Pop() == 8);
    CHECK(grid.OccupiedArea() == 2);

    std::vector<int32_t> occupied;
    for (int32_t i : grid)
    {
        occupied.push_back(i);
    }
    std::sort(occupied.begin(), occupied.end());
    CHECK(occupied == std::vector<int32_t>{grid.I(3, 4), grid.I(7, 7)});
}

TEST_CASE("PopulationGridBase::ApplyOccupied throws if the action calls Update")
{
    Hal::PopulationGrid1D<int32_t> grid(10);
    grid.Set(1, 3);
    grid.Update();
    CHECK_THROWS_AS(grid.ApplyOccupied([&](int32_t, int32_t) { grid.Update(); }), std::runtime_error);
}
