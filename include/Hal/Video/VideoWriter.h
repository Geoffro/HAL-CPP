#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace Hal
{
    // Minimal MPEG-1 video export, replacing Java's UIGrid.ToGIF with a
    // built-in video encoder rather than an external GIF/ffmpeg
    // dependency — see docs/DESIGN.md's "GIF export vs. MP4/frame-dump"
    // note. Wraps the vendored jo_mpeg encoder (third_party/jo_mpeg,
    // public domain, Jon Olick) behind a small RAII class so callers never
    // touch its C API directly.
    //
    // width/height/fps are fixed for the life of one VideoWriter, matching
    // jo_mpeg's own per-file constraints (every frame in an MPEG-1
    // sequence must share dimensions and frame rate). Quality is modest
    // (MPEG-1, no motion estimation) — a deliberate simplification to
    // avoid a heavyweight codec dependency, not a correctness issue; a
    // higher-quality encoder is a legitimate follow-up if this becomes a
    // bottleneck for actual use.
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

        // pixelsRgba must be width*height RGBA8 (4 bytes/pixel, row-major,
        // top-to-bottom); alpha is ignored by the encoder. Flushes to disk
        // before returning, so the file is inspectable/resumable-in-spirit
        // after every frame rather than only once the writer is destroyed.
        void AddFrame(const std::vector<uint8_t> &pixelsRgba);

    private:
        std::FILE *m_pFile;
        int32_t m_width;
        int32_t m_height;
        int32_t m_fps;
    };
}
