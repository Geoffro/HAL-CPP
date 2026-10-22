#include "Hal/Video/PngExport.h"

#include "Hal/Log/Logger.h"

#include "stb/stb_image_write.h"

#include <string>

namespace Hal
{
    bool ExportPng(const std::string          &path,
                   int32_t                     width,
                   int32_t                     height,
                   const std::vector<uint8_t> &pixelsRgba)
    {
        const int32_t componentsPerPixel = 4;
        size_t expectedSize = static_cast<size_t>(width) * static_cast<size_t>(height) * componentsPerPixel;
        if ((width <= 0) || (height <= 0) || (pixelsRgba.size() != expectedSize))
        {
            Logger::Instance().Log(LogSeverity::Error, "ExportPng: pixel buffer doesn't match " + std::to_string(width) + "x" + std::to_string(height) + " RGBA8 for " + path);
            return false;
        }
        int32_t strideBytes = width * componentsPerPixel;
        int result = stbi_write_png(path.c_str(), width, height, componentsPerPixel, pixelsRgba.data(), strideBytes);
        if (result == 0)
        {
            Logger::Instance().Log(LogSeverity::Error, "ExportPng failed to write: " + path);
            return false;
        }
        return true;
    }
}
