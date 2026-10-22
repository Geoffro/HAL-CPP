#include <doctest/doctest.h>

#include "TestSupport.h"

#include "Hal/Grids/GridSerialization.h"
#include "Hal/Grids/PDEGridSerialization.h"
#include "Hal/Rand.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// Save/load round trips through real files in the OS temp directory.

TEST_CASE("Hal::Rand::Save/Load reproduces the exact same future output as the unsaved generator")
{
    std::string path = TestHelpers::TempFilePath("hal_test_rand.bin");

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

    // Mid-pair, the restored sampler must return the cached second value.
    CHECK(restored.Gaussian(0.0, 1.0) == original.Gaussian(0.0, 1.0));
    CHECK(restored.Gaussian(0.0, 1.0) == original.Gaussian(0.0, 1.0));

    std::remove(path.c_str());
}

TEST_CASE("Hal::Grid2D<double> Save/Load round-trips dims, wrap flags, and field contents through a real file")
{
    std::string path = TestHelpers::TempFilePath("hal_test_grid2d.bin");

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
    std::string path = TestHelpers::TempFilePath("hal_test_pdegrid2d.bin");

    Hal::PDEGrid2D grid(4, 6, false, true);
    for (int32_t x = 0; x < 4; x++)
    {
        for (int32_t y = 0; y < 6; y++)
        {
            grid.Set(x, y, static_cast<double>(x + y));
        }
    }
    grid.Update();

    // A pending delta, so field and deltas differ when saved.
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

TEST_CASE("BinaryWriter/BinaryReader round-trip every typed value")
{
    std::string path = TestHelpers::TempFilePath("hal_test_binary_types.bin");
    {
        Hal::BinaryWriter writer(path);
        writer.WriteBool(true);
        writer.WriteInt32(-7);
        writer.WriteInt64(-9000000000LL);
        writer.WriteUint32(4000000000u);
        writer.WriteUint64(18000000000000000000ull);
        writer.WriteDouble(2.5);
        CHECK(writer.IsGood());
    }

    Hal::BinaryReader reader(path);
    CHECK(reader.ReadBool());
    CHECK(reader.ReadInt32() == -7);
    CHECK(reader.ReadInt64() == -9000000000LL);
    CHECK(reader.ReadUint32() == 4000000000u);
    CHECK(reader.ReadUint64() == 18000000000000000000ull);
    CHECK(reader.ReadDouble() == 2.5);
    CHECK(reader.IsGood());

    reader.ReadInt32();
    CHECK_FALSE(reader.IsGood());

    std::remove(path.c_str());
}

TEST_CASE("BinaryReader and BinaryWriter report a file that can't be opened")
{
    Hal::BinaryReader reader(TestHelpers::TempFilePath("hal_test_no_such_dir/missing.bin"));
    CHECK_FALSE(reader.IsOpen());
    CHECK_FALSE(reader.IsGood());

    Hal::BinaryWriter writer(TestHelpers::TempFilePath("hal_test_no_such_dir/out.bin"));
    CHECK_FALSE(writer.IsOpen());
    CHECK_FALSE(writer.IsGood());
}

TEST_CASE("Loading a grid from truncated or invalid data throws instead of allocating garbage")
{
    std::string path = TestHelpers::TempFilePath("hal_test_truncated.bin");
    {
        Hal::BinaryWriter writer(path);
        writer.WriteInt32(-3);
        writer.WriteInt32(4);
        writer.WriteBool(false);
        writer.WriteBool(false);
    }
    {
        Hal::BinaryReader reader(path);
        CHECK_THROWS_AS(Hal::LoadGrid2D<double>(reader), std::runtime_error);
    }
    {
        Hal::BinaryWriter writer(path);
        writer.WriteInt32(3);
    }
    {
        Hal::BinaryReader reader(path);
        CHECK_THROWS_AS(Hal::LoadPDEGrid1D(reader), std::runtime_error);
    }

    // A valid header followed by too few values.
    {
        Hal::BinaryWriter writer(path);
        writer.WriteInt32(2);
        writer.WriteInt32(2);
        writer.WriteBool(false);
        writer.WriteBool(false);
        writer.WriteDouble(1.0);
        writer.WriteDouble(2.0);
        writer.WriteDouble(3.0);
    }
    {
        Hal::BinaryReader reader(path);
        CHECK_THROWS_AS(Hal::LoadGrid2D<double>(reader), std::runtime_error);
    }

    // Positive dimensions whose product overflows int32_t.
    {
        Hal::BinaryWriter writer(path);
        writer.WriteInt32(1000);
        writer.WriteInt32(1000);
        writer.WriteInt32(10000);
        writer.WriteBool(false);
        writer.WriteBool(false);
        writer.WriteBool(false);
    }
    {
        Hal::BinaryReader reader(path);
        CHECK_THROWS_AS(Hal::LoadGrid3D<double>(reader), std::runtime_error);
    }
    std::remove(path.c_str());
}

TEST_CASE("Grid1D and Grid3D Save/Load round-trip dims, wrap flags, and contents")
{
    std::string path = TestHelpers::TempFilePath("hal_test_grid13.bin");
    Hal::Grid1D<int32_t> line(6, true);
    Hal::Grid3D<double> volume(2, 3, 4, false, true, false);
    for (int32_t i = 0; i < line.Length(); i++)
    {
        line.Set(i, i * i);
    }
    for (int32_t i = 0; i < volume.Length(); i++)
    {
        volume.Set(i, i * 0.5);
    }
    {
        Hal::BinaryWriter writer(path);
        Hal::SaveGrid1D(writer, line);
        Hal::SaveGrid3D(writer, volume);
    }

    Hal::BinaryReader reader(path);
    Hal::Grid1D<int32_t> loadedLine = Hal::LoadGrid1D<int32_t>(reader);
    Hal::Grid3D<double> loadedVolume = Hal::LoadGrid3D<double>(reader);
    CHECK(loadedLine.Xdim() == 6);
    CHECK(loadedLine.IsWrapX());
    CHECK(loadedLine.GetField() == line.GetField());
    CHECK(loadedVolume.Zdim() == 4);
    CHECK(loadedVolume.IsWrapY());
    CHECK_FALSE(loadedVolume.IsWrapZ());
    CHECK(loadedVolume.GetField() == volume.GetField());

    std::remove(path.c_str());
}

TEST_CASE("PDEGrid1D and PDEGrid3D Save/Load round-trip field, deltas, and update count")
{
    std::string path = TestHelpers::TempFilePath("hal_test_pdegrid13.bin");
    Hal::PDEGrid1D line(5, true);
    Hal::PDEGrid3D volume(2, 2, 3, true, false, false);
    line.SetAll(1.0);
    line.Update();
    line.Add(2, 0.25);
    volume.SetAll(2.0);
    volume.Update();
    volume.Update();
    volume.Add(1, 1, 2, -0.5);
    {
        Hal::BinaryWriter writer(path);
        Hal::SavePDEGrid1D(writer, line);
        Hal::SavePDEGrid3D(writer, volume);
    }

    Hal::BinaryReader reader(path);
    Hal::PDEGrid1D loadedLine = Hal::LoadPDEGrid1D(reader);
    Hal::PDEGrid3D loadedVolume = Hal::LoadPDEGrid3D(reader);
    CHECK(loadedLine.UpdateCt() == 1);
    CHECK(loadedLine.GetField() == line.GetField());
    CHECK(loadedLine.GetDeltas() == line.GetDeltas());
    CHECK(loadedVolume.UpdateCt() == 2);
    CHECK(loadedVolume.IsWrapX());
    CHECK(loadedVolume.GetDeltas() == volume.GetDeltas());

    std::remove(path.c_str());
}

TEST_CASE("PDEGrid SetState rejects arrays of the wrong length")
{
    Hal::PDEGrid2D grid(3, 3);
    std::vector<double> rightSize(9, 0.0);
    std::vector<double> wrongSize(8, 0.0);
    CHECK_THROWS_AS(grid.SetState(wrongSize, rightSize, 0), std::invalid_argument);
    CHECK_THROWS_AS(grid.SetState(rightSize, wrongSize, 0), std::invalid_argument);
    CHECK_NOTHROW(grid.SetState(rightSize, rightSize, 4));
    CHECK(grid.UpdateCt() == 4);
}
