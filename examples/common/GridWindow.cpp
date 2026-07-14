#include "GridWindow.h"

#include "Mp4Recorder.h"

#include "Hal/Color.h"
#include "Hal/Gpu/Window.h"
#include "Hal/Gpu/WindowRenderContext.h"
#include "Hal/Log/Logger.h"
#include "Hal/Video/PngExport.h"

#include <chrono>
#include <cmath>
#include <thread>

namespace Hal
{
    namespace Examples
    {
        GridWindow::GridWindow(const std::string &title,
                               int32_t            xDim,
                               int32_t            yDim,
                               int32_t            scaleFactor,
                               bool               active)
            : m_xDim(xDim),
              m_yDim(yDim),
              m_scaleFactor(scaleFactor),
              m_active(active),
              m_pixels(static_cast<size_t>(xDim) * static_cast<size_t>(yDim) * 4, 0)
        {
            if (m_active)
            {
                m_window = std::make_unique<Gpu::Window>(xDim * scaleFactor, yDim * scaleFactor, title);
                m_render = std::make_unique<Gpu::WindowRenderContext>(*m_window, xDim, yDim);
            }
        }

        GridWindow::GridWindow(int32_t xDim,
                               int32_t yDim,
                               int32_t scaleFactor)
            : GridWindow("HAL", xDim, yDim, scaleFactor, true)
        {
        }

        GridWindow::GridWindow(int32_t xDim,
                               int32_t yDim)
            : GridWindow("HAL", xDim, yDim, 1, true)
        {
        }

        GridWindow::~GridWindow()
        {
            StopRecording();
        }

        void GridWindow::PaintCell(int32_t x,
                                   int32_t y,
                                   int32_t color)
        {
            // Flip y so buffer row 0 is the top of the image (y == 0 bottom).
            int32_t imageRow = m_yDim - 1 - y;
            size_t  base = (static_cast<size_t>(imageRow) * static_cast<size_t>(m_xDim) + static_cast<size_t>(x)) * 4;
            m_pixels[base + 0] = static_cast<uint8_t>(GetRed256(color));
            m_pixels[base + 1] = static_cast<uint8_t>(GetGreen256(color));
            m_pixels[base + 2] = static_cast<uint8_t>(GetBlue256(color));
            m_pixels[base + 3] = static_cast<uint8_t>(GetAlpha256(color));
        }

        void GridWindow::SetPix(int32_t x,
                                int32_t y,
                                int32_t color)
        {
            if ((x < 0) || (x >= m_xDim) || (y < 0) || (y >= m_yDim))
            {
                return;
            }
            PaintCell(x, y, color);
        }

        void GridWindow::SetPix(int32_t i,
                                int32_t color)
        {
            SetPix(i / m_yDim, i % m_yDim, color);
        }

        int32_t GridWindow::GetPix(int32_t x,
                                   int32_t y) const
        {
            if ((x < 0) || (x >= m_xDim) || (y < 0) || (y >= m_yDim))
            {
                return 0;
            }
            int32_t imageRow = m_yDim - 1 - y;
            size_t  base = (static_cast<size_t>(imageRow) * static_cast<size_t>(m_xDim) + static_cast<size_t>(x)) * 4;
            return Rgba256(m_pixels[base + 0], m_pixels[base + 1], m_pixels[base + 2], m_pixels[base + 3]);
        }

        void GridWindow::Clear(int32_t color)
        {
            for (int32_t x = 0; x < m_xDim; x++)
            {
                for (int32_t y = 0; y < m_yDim; y++)
                {
                    PaintCell(x, y, color);
                }
            }
        }

        void GridWindow::SetPixels(const std::vector<uint8_t> &rgba)
        {
            if (rgba.size() == m_pixels.size())
            {
                m_pixels = rgba;
            }
        }

        void GridWindow::Circle(double  x,
                                double  y,
                                double  radius,
                                int32_t color)
        {
            // Rasterize the disc using pixel centers, at least one cell so a
            // tiny radius still shows up.
            double  r = (radius < 0.5) ? 0.5 : radius;
            double  rSq = r * r;
            int32_t xLo = static_cast<int32_t>(std::floor(x - r));
            int32_t xHi = static_cast<int32_t>(std::ceil(x + r));
            int32_t yLo = static_cast<int32_t>(std::floor(y - r));
            int32_t yHi = static_cast<int32_t>(std::ceil(y + r));
            for (int32_t px = xLo; px <= xHi; px++)
            {
                for (int32_t py = yLo; py <= yHi; py++)
                {
                    double dx = (px + 0.5) - x;
                    double dy = (py + 0.5) - y;
                    if ((dx * dx + dy * dy) <= rSq)
                    {
                        SetPix(px, py, color);
                    }
                }
            }
        }

        void GridWindow::Tick()
        {
            if (m_active)
            {
                m_window->PollEvents();
                m_render->UpdateTexture(m_pixels);
                m_render->DrawFrame();
            }
            EmitFrame();
        }

        void GridWindow::TickPause(int32_t millis)
        {
            Tick();
            if (millis > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(millis));
            }
        }

        void *GridWindow::NativeHandle() const
        {
            if (!m_active)
            {
                return nullptr;
            }
            return static_cast<void *>(m_window->Handle());
        }

        bool GridWindow::IsClosed() const
        {
            if (!m_active)
            {
                return false;
            }
            return m_window->ShouldClose();
        }

        bool GridWindow::ToPng(const std::string &path) const
        {
            return ExportPng(path, m_xDim, m_yDim, m_pixels);
        }

        void GridWindow::StartRecording(const std::string &path,
                                        int32_t            fps)
        {
            // Upscale the magnified grid to the on-screen size (crisp
            // nearest-neighbor). The recorder handles the ffmpeg-vs-jo_mpeg
            // choice and the .mpg fallback naming.
            int32_t scale = (m_scaleFactor > 1) ? m_scaleFactor : 1;
            m_recorder = std::make_unique<Mp4Recorder>(path, m_xDim, m_yDim, fps, scale);
            if (!m_recorder->IsOpen())
            {
                m_recorder.reset();
            }
        }

        void GridWindow::EmitFrame()
        {
            if (m_recorder && m_recorder->IsOpen())
            {
                m_recorder->AddFrame(m_pixels);
            }
        }

        void GridWindow::RecordFrame()
        {
            EmitFrame();
        }

        void GridWindow::StopRecording()
        {
            m_recorder.reset();
        }
    }
}
