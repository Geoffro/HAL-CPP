#include "Hal/Video/VideoWriter.h"

#include "Hal/Log/Logger.h"

// jo_mpeg's own single-header convention: defining
// JO_MPEG_HEADER_FILE_ONLY before including it gives just the
// jo_write_mpeg() declaration, not its (non-static, extern-linkage)
// implementation — so this translation unit can call it without also
// defining it, avoiding a duplicate-symbol clash with
// JoMpegImpl.cpp (the one place that does compile the implementation).
#define JO_MPEG_HEADER_FILE_ONLY
#include "jo_mpeg/jo_mpeg.h"
#undef JO_MPEG_HEADER_FILE_ONLY

namespace Hal
{
    VideoWriter::VideoWriter(const std::string &path,
                             int32_t            width,
                             int32_t            height,
                             int32_t            fps)
        : m_pFile(std::fopen(path.c_str(), "wb")),
          m_width(width),
          m_height(height),
          m_fps(fps)
    {
        if (m_pFile == nullptr)
        {
            Logger::Instance().Log(LogSeverity::Error, "VideoWriter failed to open file for writing: " + path);
        }
    }

    VideoWriter::~VideoWriter()
    {
        if (m_pFile != nullptr)
        {
            std::fclose(m_pFile);
        }
    }

    bool VideoWriter::IsOpen() const
    {
        return m_pFile != nullptr;
    }

    void VideoWriter::AddFrame(const std::vector<uint8_t> &pixelsRgba)
    {
        if (m_pFile == nullptr)
        {
            return;
        }
        jo_write_mpeg(m_pFile, pixelsRgba.data(), m_width, m_height, m_fps);
        std::fflush(m_pFile);
    }
}
