#include <doctest/doctest.h>

#include "Hal/Grids/GridSerialization.h"
#include "Hal/Grids/PDEGridSerialization.h"
#include "Hal/Rand.h"

#include <cstdio>
#include <filesystem>
#include <string>

// ---------------------------------------------------------------------
// Phase 5: save/load round-trip tests. Each writes to a real file under
// the OS temp directory (not an in-memory stream) since the point of
// this phase is persisting to disk between process runs, then removes
// it afterward so repeated test runs don't accumulate files.
// ---------------------------------------------------------------------

static std::string TempFilePath(const std::string &name)
{
    return (std::filesystem::temp_directory_path() / name).string();
}

TEST_CASE("Hal::Rand::Save/Load reproduces the exact same future output as the unsaved generator")
{
    std::string path = TempFilePath("hal_test_rand.bin");

    Hal::Rand original(12345);
    for (int32_t i = 0; i < 50; i++)
    {
        original.Double();
    }

    {
        Hal::BinaryWriter writer(path);
        REQUIRE(writer.IsOpen());
        original.Save(writer);
    }

    Hal::Rand restored(0);
    {
        Hal::BinaryReader reader(path);
        REQUIRE(reader.IsOpen());
        restored.Load(reader);
    }

    for (int32_t i = 0; i < 100; i++)
    {
        CAPTURE(i);
        CHECK(restored.Double() == original.Double());
    }

    // A restored sampler mid-Gaussian-pair must return the cached
    // companion value, not silently redraw both values of the pair.
    CHECK(restored.Gaussian(0.0, 1.0) == original.Gaussian(0.0, 1.0));
    CHECK(restored.Gaussian(0.0, 1.0) == original.Gaussian(0.0, 1.0));

    std::remove(path.c_str());
}

TEST_CASE("Hal::Grid2D<double> Save/Load round-trips dims, wrap flags, and field contents through a real file")
{
    std::string path = TempFilePath("hal_test_grid2d.bin");

    Hal::Grid2D<double> grid(5, 7, true, false);
    for (int32_t x = 0; x < 5; x++)
    {
        for (int32_t y = 0; y < 7; y++)
        {
            grid.Set(x, y, static_cast<double>(x * 100 + y));
        }
    }

    {
        Hal::BinaryWriter writer(path);
        REQUIRE(writer.IsOpen());
        Hal::SaveGrid2D(writer, grid);
    }

    Hal::Grid2D<double> loaded = [&]()
    {
        Hal::BinaryReader reader(path);
        REQUIRE(reader.IsOpen());
        return Hal::LoadGrid2D<double>(reader);
    }();

    CHECK(loaded.Xdim() == 5);
    CHECK(loaded.Ydim() == 7);
    CHECK(loaded.IsWrapX() == true);
    CHECK(loaded.IsWrapY() == false);
    for (int32_t x = 0; x < 5; x++)
    {
        for (int32_t y = 0; y < 7; y++)
        {
            CAPTURE(x);
            CAPTURE(y);
            CHECK(loaded.Get(x, y) == doctest::Approx(x * 100 + y));
        }
    }

    std::remove(path.c_str());
}

TEST_CASE("Hal::PDEGrid2D Save/Load round-trips field, pending deltas, and updateCt through a real file")
{
    std::string path = TempFilePath("hal_test_pdegrid2d.bin");

    Hal::PDEGrid2D grid(4, 6, false, true);
    for (int32_t x = 0; x < 4; x++)
    {
        for (int32_t y = 0; y < 6; y++)
        {
            grid.Set(x, y, static_cast<double>(x + y));
        }
    }
    grid.Update();

    // Leave a pending (not-yet-committed) delta before saving, so the
    // round-trip is checked at a point where field and deltas actually
    // differ from each other.
    grid.Add(2, 3, 0.5);

    {
        Hal::BinaryWriter writer(path);
        REQUIRE(writer.IsOpen());
        Hal::SavePDEGrid2D(writer, grid);
    }

    Hal::PDEGrid2D loaded = [&]()
    {
        Hal::BinaryReader reader(path);
        REQUIRE(reader.IsOpen());
        return Hal::LoadPDEGrid2D(reader);
    }();

    CHECK(loaded.Xdim() == 4);
    CHECK(loaded.Ydim() == 6);
    CHECK(loaded.IsWrapX() == false);
    CHECK(loaded.IsWrapY() == true);
    CHECK(loaded.UpdateCt() == grid.UpdateCt());
    for (int32_t x = 0; x < 4; x++)
    {
        for (int32_t y = 0; y < 6; y++)
        {
            CAPTURE(x);
            CAPTURE(y);
            CHECK(loaded.Get(x, y) == doctest::Approx(grid.Get(x, y)));
            CHECK(loaded.GetDeltas()[grid.I(x, y)] == doctest::Approx(grid.GetDeltas()[grid.I(x, y)]));
        }
    }

    std::remove(path.c_str());
}
