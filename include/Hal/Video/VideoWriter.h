#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace Hal
{
    // MPEG-1 video writer over the vendored jo_mpeg encoder (no external
    // codec dependency; modest quality, no motion estimation). Size and
    // frame rate are fixed for the file's lifetime.
    class VideoWriter
    {
    public:
        VideoWriter(const std::string &path,
                    int32_t            width,
                    int32_t            height,
                    int32_t            fps);
        ~VideoWriter();

        VideoWriter(const VideoWriter &other) = delete;
        VideoWriter &operator=(const VideoWriter &other) = delete;

        bool IsOpen() const;

        // pixelsRgba must be width*height RGBA8 (row 0 at the top); alpha is
        // ignored. Flushes to disk after each frame. Returns false if the
        // file isn't open (the constructor logged that) and logs and returns
        // false for a wrong-size buffer or a failed write.
        bool AddFrame(const std::vector<uint8_t> &pixelsRgba);

    private:
        std::FILE *m_pFile;
        int32_t m_width;
        int32_t m_height;
        int32_t m_fps;
    };
}
