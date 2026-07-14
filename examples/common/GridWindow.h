#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Hal
{
    namespace Gpu
    {
        class Window;
        class WindowRenderContext;
    }

    namespace Examples
    {
        class Mp4Recorder;

        // A drop-in analogue of HAL's Gui/GridWindow (a UIGrid backed by a
        // window): a scaleFactor-magnified pixel grid you paint one cell at a
        // time with SetPix, then present with TickPause. It mirrors the Java
        // API closely so the ported example models read almost identically.
        //
        // With active == true it opens a real Vulkan/GLFW window and TickPause
        // uploads the pixel buffer and draws a frame. With active == false no
        // window is created at all (nothing to display, e.g. on a headless
        // run) — SetPix/Clear still populate the buffer, TickPause becomes a
        // no-op, and you capture output with ToPng / RecordFrame instead. This
        // is the same active/headless split Java's `active` constructor flag
        // provides, and lets a model be flipped between a live window and a
        // PNG/MP4 dump by toggling one argument.
        //
        // Coordinates follow HAL's convention: x in [0,xDim), y in [0,yDim)
        // with y == 0 the BOTTOM row. Colors are packed ARGB int32_t values
        // (see Hal/Color.h). The stored image is flipped so both the on-screen
        // texture and exported PNGs come out right-side-up.
        class GridWindow
        {
        public:
            GridWindow(const std::string &title,
                       int32_t            xDim,
                       int32_t            yDim,
                       int32_t            scaleFactor,
                       bool               active);

            GridWindow(int32_t xDim,
                       int32_t yDim,
                       int32_t scaleFactor);

            GridWindow(int32_t xDim,
                       int32_t yDim);

            ~GridWindow();

            GridWindow(const GridWindow &other) = delete;
            GridWindow &operator=(const GridWindow &other) = delete;

            // Paint a single cell (ARGB). Out-of-range coordinates are ignored.
            void SetPix(int32_t x,
                        int32_t y,
                        int32_t color);

            // Flattened index form: i == x * yDim + y (HAL's row-major order).
            void SetPix(int32_t i,
                        int32_t color);

            int32_t GetPix(int32_t x,
                           int32_t y) const;

            void Clear(int32_t color);

            // Replace the whole pixel buffer with a pre-composited RGBA8
            // image (row-major, top-to-bottom, size xDim*yDim*4) — e.g. the
            // output of MultiWellExperiment::Pixels(). No y-flip is applied
            // (the buffer is already image-oriented).
            void SetPixels(const std::vector<uint8_t> &rgba);

            // Draw a filled disc centered at continuous coords (x, y) with the
            // given radius, all in grid/cell units (one cell == one texture
            // pixel). The C++ analogue of OpenGL2DWindow.Circle for off-lattice
            // agents drawn at their real positions.
            void Circle(double  x,
                        double  y,
                        double  radius,
                        int32_t color);

            // Present the current buffer and pace the loop to ~millis. On a
            // headless window this only paces (and is usually passed 0). Also
            // captures a video frame if RecordFrame recording is active.
            void TickPause(int32_t millis);

            // Present without pacing (upload + draw + poll events).
            void Tick();

            bool IsClosed() const;

            int32_t Xdim() const { return m_xDim; }
            int32_t Ydim() const { return m_yDim; }

            bool IsActive() const { return m_active; }

            // The underlying GLFWwindow* (as void*) for input polling, or
            // nullptr on a headless window. Cast to GLFWwindow* at the call
            // site (keeps GLFW out of this header).
            void *NativeHandle() const;

            // Write the current buffer to a PNG file.
            bool ToPng(const std::string &path) const;

            // Begin/append/end a recording of presented frames. Each
            // TickPause (or explicit RecordFrame) appends the current
            // buffer. If ffmpeg is on the PATH the file is a real H.264
            // MP4 (QuickTime-playable, upscaled by scaleFactor with
            // nearest-neighbor to keep the pixel-art look); otherwise it
            // falls back to the built-in jo_mpeg encoder and the output is
            // renamed to .mpg, since that stream is MPEG-1 (playable in
            // VLC/ffmpeg, not modern QuickTime) — see docs/DESIGN.md.
            void StartRecording(const std::string &path,
                                int32_t            fps);
            void RecordFrame();
            void StopRecording();

        private:
            void PaintCell(int32_t x,
                           int32_t y,
                           int32_t color);

            // Append the current buffer to the recorder, if one is open.
            void EmitFrame();

            int32_t                                   m_xDim;
            int32_t                                   m_yDim;
            int32_t                                   m_scaleFactor;
            bool                                      m_active;
            std::vector<uint8_t>                      m_pixels; // RGBA8, y-flipped
            std::unique_ptr<Gpu::Window>              m_window;
            std::unique_ptr<Gpu::WindowRenderContext> m_render;
            std::unique_ptr<Mp4Recorder>              m_recorder;
        };
    }
}
