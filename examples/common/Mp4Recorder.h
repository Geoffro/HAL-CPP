#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace Hal
{
    class VideoWriter;

    namespace Examples
    {
        // Records a stream of RGBA8 frames to a video file. If ffmpeg is on
        // the PATH the output is a real QuickTime-playable H.264 .mp4
        // (optionally integer-upscaled with nearest-neighbor to keep a
        // pixel-art look crisp); otherwise it falls back to the built-in
        // jo_mpeg encoder (Hal::VideoWriter), whose stream is MPEG-1 and so
        // is written to a correctly named .mpg (modern QuickTime can't play
        // MPEG-1; VLC/ffmpeg can). Shared by GridWindow and the 3D showcase
        // so the ffmpeg-pipe logic lives in exactly one place — see
        // docs/DESIGN.md's video-export note.
        //
        // Frames must all be width*height*4 bytes (row-major, top-to-bottom).
        class Mp4Recorder
        {
        public:
            // upscale multiplies the stored resolution in the encoded video
            // (>= 1); pass 1 for no scaling.
            Mp4Recorder(const std::string &path,
                        int32_t            width,
                        int32_t            height,
                        int32_t            fps,
                        int32_t            upscale);
            ~Mp4Recorder();

            Mp4Recorder(const Mp4Recorder &other) = delete;
            Mp4Recorder &operator=(const Mp4Recorder &other) = delete;

            bool IsOpen() const;

            void AddFrame(const std::vector<uint8_t> &rgba);

        private:
            bool TryStartFfmpeg(const std::string &path,
                                int32_t            fps,
                                int32_t            upscale);

            int32_t                      m_width;
            int32_t                      m_height;
            std::FILE                   *m_pFfmpeg;
            std::unique_ptr<VideoWriter> m_fallback;
        };
    }
}
