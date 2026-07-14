#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Hal
{
    // Writes a width*height RGBA8 pixel buffer (4 bytes/pixel, row-major,
    // top-to-bottom — row 0 is the top of the image) to a PNG file at
    // path, via the vendored stb_image_write (third_party/stb). Returns
    // false (and logs via Hal::Logger) on failure rather than throwing —
    // an export failure shouldn't crash a long-running simulation.
    bool ExportPng(const std::string          &path,
                   int32_t                     width,
                   int32_t                     height,
                   const std::vector<uint8_t> &pixelsRgba);
}
