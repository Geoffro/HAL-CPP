#pragma once

#include "Hal/Color.h"
#include "Hal/Grids/Grid2D.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    // Renders a scalar Grid2D<T> to a width*height RGBA8 pixel buffer (row-
    // major, top-to-bottom) suitable for ExportPng/VideoWriter, via a
    // caller-supplied value-to-color function returning a packed ARGB
    // int32_t (Hal::Color.h's format) — the C++ equivalent of populating
    // Java's UIGrid pixel buffer one cell at a time before ToPNG()/ToGIF().
    //
    // Grid y=0 is conventionally the bottom row (matching HAL's own
    // Util.PNGtoGrid, which flips on load "so picture is right-side-up"),
    // while image row 0 is the top, so this flips y when writing pixels —
    // without it, exported images would come out upside down relative to
    // how the grid's own coordinate system is normally reasoned about.
    template <typename T, typename ColorFn>
    std::vector<uint8_t> RenderGrid2DToRgba(const Grid2D<T> &grid,
                                           ColorFn           colorFn)
    {
        int32_t width = grid.Xdim();
        int32_t height = grid.Ydim();
        std::vector<uint8_t> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);

        for (int32_t x = 0; x < width; x++)
        {
            for (int32_t y = 0; y < height; y++)
            {
                int32_t argb = colorFn(grid.Get(x, y));
                int32_t imageRow = height - 1 - y;
                size_t pixelIndex = (static_cast<size_t>(imageRow) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4;
                pixels[pixelIndex + 0] = static_cast<uint8_t>(GetRed256(argb));
                pixels[pixelIndex + 1] = static_cast<uint8_t>(GetGreen256(argb));
                pixels[pixelIndex + 2] = static_cast<uint8_t>(GetBlue256(argb));
                pixels[pixelIndex + 3] = static_cast<uint8_t>(GetAlpha256(argb));
            }
        }
        return pixels;
    }
}
