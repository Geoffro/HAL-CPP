#include "Hal/Video/PngExport.h"

#include "Hal/Log/Logger.h"

#include "stb/stb_image_write.h"

namespace Hal
{
    bool ExportPng(const std::string          &path,
                   int32_t                     width,
                   int32_t                     height,
                   const std::vector<uint8_t> &pixelsRgba)
    {
        const int32_t componentsPerPixel = 4;
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
