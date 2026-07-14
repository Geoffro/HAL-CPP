#include <doctest/doctest.h>

#include "Hal/Grids/Grid2D.h"
#include "Hal/Video/GridRender.h"
#include "Hal/Video/PngExport.h"
#include "Hal/Video/VideoWriter.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

// ---------------------------------------------------------------------
// PNG export is checked against an exact ground truth: PNG is lossless,
// so decoding the file back (via the separately-vendored stb_image
// reader, not the writer used to produce it) must reproduce every byte
// of the original RGBA buffer exactly — not just "looks like an image."
//
// MPEG video export can't be checked this way (it's lossy, and HAL
// doesn't vendor a decoder) — that test instead checks the small set of
// things that are true regardless of compression: the file starts with
// the correct MPEG sequence-header magic bytes, ends with the
// sequence-end code, and grows as more frames are added.
// ---------------------------------------------------------------------

static std::string TempFilePath(const std::string &name)
{
    return (std::filesystem::temp_directory_path() / name).string();
}

TEST_CASE("Hal::ExportPng + Hal::RenderGrid2DToRgba round-trips exactly through a real PNG file")
{
    std::string path = TempFilePath("hal_test_export.png");

    Hal::Grid2D<double> grid(6, 4);
    for (int32_t x = 0; x < 6; x++)
    {
        for (int32_t y = 0; y < 4; y++)
        {
            grid.Set(x, y, static_cast<double>(x) / 5.0);
        }
    }

    auto colorFn = [](double value)
    {
        return Hal::Rgb(value, 0.25, 1.0 - value);
    };

    std::vector<uint8_t> pixels = Hal::RenderGrid2DToRgba(grid, colorFn);
    REQUIRE(Hal::ExportPng(path, grid.Xdim(), grid.Ydim(), pixels));

    int decodedWidth = 0;
    int decodedHeight = 0;
    int decodedChannels = 0;
    unsigned char *pDecoded = stbi_load(path.c_str(), &decodedWidth, &decodedHeight, &decodedChannels, 4);
    REQUIRE(pDecoded != nullptr);
    CHECK(decodedWidth == grid.Xdim());
    CHECK(decodedHeight == grid.Ydim());

    for (size_t i = 0; i < pixels.size(); i++)
    {
        CAPTURE(i);
        CHECK(static_cast<int>(pDecoded[i]) == static_cast<int>(pixels[i]));
    }

    stbi_image_free(pDecoded);
    std::remove(path.c_str());
}

TEST_CASE("Hal::VideoWriter produces a file with a valid MPEG sequence header and end code, growing with each frame")
{
    std::string path = TempFilePath("hal_test_export.mpg");

    const int32_t width = 16;
    const int32_t height = 16;
    std::vector<uint8_t> frame(static_cast<size_t>(width) * height * 4, 0);

    std::vector<size_t> sizeAfterEachFrame;
    {
        Hal::VideoWriter writer(path, width, height, 30);
        REQUIRE(writer.IsOpen());
        for (int32_t frameIndex = 0; frameIndex < 3; frameIndex++)
        {
            for (size_t i = 0; i < frame.size(); i += 4)
            {
                frame[i] = static_cast<uint8_t>((frameIndex * 40 + i) % 256);
            }
            writer.AddFrame(frame);
            sizeAfterEachFrame.push_back(static_cast<size_t>(std::filesystem::file_size(path)));
        }
    }

    REQUIRE(sizeAfterEachFrame.size() == 3);
    CHECK(sizeAfterEachFrame[0] > 0);
    CHECK(sizeAfterEachFrame[1] > sizeAfterEachFrame[0]);
    CHECK(sizeAfterEachFrame[2] > sizeAfterEachFrame[1]);

    std::ifstream file(path, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    REQUIRE(bytes.size() >= 8);

    // MPEG-1 sequence header start code.
    CHECK(bytes[0] == 0x00);
    CHECK(bytes[1] == 0x00);
    CHECK(bytes[2] == 0x01);
    CHECK(bytes[3] == 0xB3);

    // Sequence-end code, written once by the last AddFrame() call.
    size_t n = bytes.size();
    CHECK(bytes[n - 4] == 0x00);
    CHECK(bytes[n - 3] == 0x00);
    CHECK(bytes[n - 2] == 0x01);
    CHECK(bytes[n - 1] == 0xB7);

    std::remove(path.c_str());
}
