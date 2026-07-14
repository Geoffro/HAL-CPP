#include "Mp4Recorder.h"

#include "Hal/Log/Logger.h"
#include "Hal/Video/VideoWriter.h"

#include <cstdlib>

namespace Hal
{
    namespace Examples
    {
        Mp4Recorder::Mp4Recorder(const std::string &path,
                                 int32_t            width,
                                 int32_t            height,
                                 int32_t            fps,
                                 int32_t            upscale)
            : m_width(width),
              m_height(height),
              m_pFfmpeg(nullptr),
              m_fallback()
        {
            int32_t scale = (upscale > 1) ? upscale : 1;
            if (TryStartFfmpeg(path, fps, scale))
            {
                return;
            }

            // Fallback: the built-in jo_mpeg encoder emits an MPEG-1
            // elementary stream, which modern QuickTime can't play and which
            // is not an MP4 container — name it honestly .mpg.
            std::string fallbackPath = path;
            size_t dot = fallbackPath.find_last_of('.');
            if (dot != std::string::npos)
            {
                fallbackPath = fallbackPath.substr(0, dot);
            }
            fallbackPath += ".mpg";

            m_fallback = std::make_unique<VideoWriter>(fallbackPath, width, height, fps);
            if (!m_fallback->IsOpen())
            {
                Logger::Instance().Log(LogSeverity::Error, "Mp4Recorder: failed to open video file " + fallbackPath);
                m_fallback.reset();
            }
            else
            {
                Logger::Instance().Log(LogSeverity::Info, "Mp4Recorder: ffmpeg not found; wrote MPEG-1 " + fallbackPath + " (play with VLC/ffmpeg, not QuickTime)");
            }
        }

        Mp4Recorder::~Mp4Recorder()
        {
            if (m_pFfmpeg != nullptr)
            {
                pclose(m_pFfmpeg);
                m_pFfmpeg = nullptr;
            }
            m_fallback.reset();
        }

        bool Mp4Recorder::TryStartFfmpeg(const std::string &path,
                                         int32_t            fps,
                                         int32_t            upscale)
        {
            // Cheap availability probe; command -v exits 0 iff ffmpeg is on
            // the PATH.
            if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0)
            {
                return false;
            }

            int32_t outW = m_width * upscale;
            int32_t outH = m_height * upscale;

            // Read raw RGBA frames from stdin, upscale by an integer factor
            // with nearest-neighbor (crisp pixels), force even dimensions,
            // and encode QuickTime-friendly H.264 (yuv420p + faststart).
            std::string command = "ffmpeg -y -loglevel error -f rawvideo -pixel_format rgba"
                                  " -video_size " + std::to_string(m_width) + "x" + std::to_string(m_height) +
                                  " -framerate " + std::to_string(fps) +
                                  " -i - -an -vf \"scale=" + std::to_string(outW) + ":" + std::to_string(outH) +
                                  ":flags=neighbor,pad=ceil(iw/2)*2:ceil(ih/2)*2\""
                                  " -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \"" + path + "\"";

            m_pFfmpeg = popen(command.c_str(), "w");
            if (m_pFfmpeg == nullptr)
            {
                return false;
            }
            Logger::Instance().Log(LogSeverity::Info, "Mp4Recorder: recording H.264 to " + path + " via ffmpeg (" + std::to_string(outW) + "x" + std::to_string(outH) + ")");
            return true;
        }

        bool Mp4Recorder::IsOpen() const
        {
            return (m_pFfmpeg != nullptr) || (m_fallback && m_fallback->IsOpen());
        }

        void Mp4Recorder::AddFrame(const std::vector<uint8_t> &rgba)
        {
            if (m_pFfmpeg != nullptr)
            {
                std::fwrite(rgba.data(), 1, rgba.size(), m_pFfmpeg);
                return;
            }
            if (m_fallback && m_fallback->IsOpen())
            {
                m_fallback->AddFrame(rgba);
            }
        }
    }
}
