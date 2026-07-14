#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GLFWwindow;

namespace Hal
{
    namespace Gpu
    {
        // Mouse buttons for Window::MouseButtonDown, so callers polling
        // input through Window don't need GLFW's own constants (keeps
        // GLFW out of this header).
        enum class MouseButton : int32_t
        {
            Left   = 0,
            Right  = 1,
            Middle = 2
        };

        // Thin RAII wrapper over a single GLFW window, the foundation
        // the Vulkan graphics pipeline is built on top of — replacing
        // Java's Swing-based
        // `UIWindow`/`OpenGL2DWindow`/`OpenGL3DWindow`. Owns GLFW's global
        // init/terminate lifecycle via a process-wide reference count (so
        // multiple Window instances, though unusual for a HAL model, don't
        // double-init or tear down GLFW out from under each other).
        //
        // Deliberately doesn't own any Vulkan objects itself — creating a
        // VkSurfaceKHR needs a VkInstance, which doesn't exist until
        // WindowRenderContext constructs one (partly informed by this
        // window's own required-extensions list), so that ordering lives
        // in WindowRenderContext instead.
        class Window
        {
        public:
            Window(int32_t            width,
                   int32_t            height,
                   const std::string &title);
            ~Window();

            Window(const Window &other) = delete;
            Window &operator=(const Window &other) = delete;

            bool ShouldClose() const;
            void PollEvents();

            void FramebufferSize(int32_t &outWidth,
                                  int32_t &outHeight) const;

            // Mouse/scroll polling for camera controls (e.g. the hardware
            // Scene3D renderer's orbit camera) without the caller touching
            // GLFW directly. Cursor coordinates are in screen (window)
            // pixels, origin top-left, matching GLFW's convention.
            void CursorPos(double &outX,
                           double &outY) const;

            bool MouseButtonDown(MouseButton button) const;

            // Vertical scroll-wheel movement accumulated since the last
            // call (GLFW only reports scrolling through a callback, so the
            // window buffers it between polls). Positive is "scroll up".
            double ConsumeScrollY();

            // True once if the framebuffer was resized since the last
            // call (reported by GLFW's resize callback and buffered the
            // same way as scrolling). The render context polls this each
            // frame to know the swapchain needs recreating — resize
            // events on some platforms never surface as a Vulkan
            // OUT_OF_DATE result, so acquire/present results alone are
            // not a reliable signal.
            bool ConsumeFramebufferResized();

            GLFWwindow *Handle() const;

            // Vulkan instance extensions GLFW needs enabled to create a
            // surface for any window (queried once via
            // glfwGetRequiredInstanceExtensions, independent of any
            // particular Window instance).
            static std::vector<const char *> RequiredInstanceExtensions();

        private:
            static void ScrollCallback(GLFWwindow *pWindow,
                                       double      xOffset,
                                       double      yOffset);

            static void FramebufferSizeCallback(GLFWwindow *pWindow,
                                                int32_t     width,
                                                int32_t     height);

            GLFWwindow *m_pHandle;
            double      m_scrollY;
            bool        m_framebufferResized;
        };
    }
}
