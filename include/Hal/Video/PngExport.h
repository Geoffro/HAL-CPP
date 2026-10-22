#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Hal
{
    // Writes a width*height RGBA8 buffer (row 0 at the top) to a PNG file.
    // Returns false and logs on failure, including a buffer that isn't
    // width*height*4 bytes.
    bool ExportPng(const std::string          &path,
                   int32_t                     width,
                   int32_t                     height,
                   const std::vector<uint8_t> &pixelsRgba);
}
