#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GLFWwindow;

namespace Hal
{
    namespace Gpu
    {
        // Mouse buttons for MouseButtonDown, keeping GLFW out of this header.
        enum class MouseButton : int32_t
        {
            Left   = 0,
            Right  = 1,
            Middle = 2
        };

        // RAII wrapper over a GLFW window. GLFW's global init/terminate is
        // reference-counted across Window instances. Vulkan surface creation is
        // left to the render contexts, which own the instance.
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

            // Cursor position in window pixels, origin top-left.
            void CursorPos(double &outX,
                           double &outY) const;

            bool MouseButtonDown(MouseButton button) const;

            // Vertical scroll accumulated since the last call; positive is up.
            double ConsumeScrollY();

            // True once after each framebuffer resize. Render contexts poll this
            // because some platforms never report a resize as OUT_OF_DATE.
            bool ConsumeFramebufferResized();

            GLFWwindow *Handle() const;

            // Vulkan instance extensions GLFW needs to create window surfaces.
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
