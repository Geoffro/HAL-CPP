#pragma once

#include "Hal/Color.h"
#include "Hal/Grids/Grid2D.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    // Renders a Grid2D<T> to a width*height RGBA8 buffer for ExportPng or
    // VideoWriter, mapping each value through colorFn to a packed ARGB
    // color. Grid y is up, so grid y=0 becomes the bottom image row.
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
