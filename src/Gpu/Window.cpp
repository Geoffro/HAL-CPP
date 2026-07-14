#include "Hal/Gpu/Window.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Log/Logger.h"

#include <volk.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <stdexcept>

namespace Hal
{
    namespace Gpu
    {
        namespace Internal
        {
            int32_t g_windowRefCount = 0;
        }

        Window::Window(int32_t            width,
                       int32_t            height,
                       const std::string &title)
            : m_pHandle(nullptr),
              m_scrollY(0.0),
              m_framebufferResized(false)
        {
            if (Internal::g_windowRefCount == 0)
            {
                // glfwInitVulkanLoader sets an *init hint*, which glfwInit()
                // reads once and then discards — it has no effect if called
                // any later. It must run first so GLFW uses the same
                // vkGetInstanceProcAddr volk already resolved, instead of
                // dlopen-ing its own copy of the Vulkan loader (which can
                // fail to find a loader installed somewhere not on GLFW's
                // own search path, e.g. Homebrew's /usr/local/lib on macOS,
                // even though volk found it fine).
                Internal::CheckResult(volkInitialize(), "volkInitialize failed — is the Vulkan loader installed?");
                glfwInitVulkanLoader(vkGetInstanceProcAddr);

                if (glfwInit() == GLFW_FALSE)
                {
                    Logger::Instance().Log(LogSeverity::Error, "glfwInit failed — no usable windowing system in this environment?");
                    throw std::runtime_error("glfwInit failed");
                }
            }
            Internal::g_windowRefCount++;

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            m_pHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
            if (m_pHandle == nullptr)
            {
                Logger::Instance().Log(LogSeverity::Error, "glfwCreateWindow failed to create a window titled \"" + title + "\"");
                throw std::runtime_error("glfwCreateWindow failed");
            }

            // Scroll and resize events only arrive through callbacks, so
            // buffer them on the Window for ConsumeScrollY/
            // ConsumeFramebufferResized. imgui_impl_glfw (wired in by
            // WindowGuiRenderer, after this constructor) chains any
            // previously installed callbacks, so both can coexist.
            glfwSetWindowUserPointer(m_pHandle, this);
            glfwSetScrollCallback(m_pHandle, ScrollCallback);
            glfwSetFramebufferSizeCallback(m_pHandle, FramebufferSizeCallback);
        }

        void Window::FramebufferSizeCallback(GLFWwindow *pWindow,
                                             int32_t     width,
                                             int32_t     height)
        {
            (void)width;
            (void)height;
            Window *pSelf = static_cast<Window *>(glfwGetWindowUserPointer(pWindow));
            if (pSelf != nullptr)
            {
                pSelf->m_framebufferResized = true;
            }
        }

        void Window::ScrollCallback(GLFWwindow *pWindow,
                                    double      xOffset,
                                    double      yOffset)
        {
            (void)xOffset;
            Window *pSelf = static_cast<Window *>(glfwGetWindowUserPointer(pWindow));
            if (pSelf != nullptr)
            {
                pSelf->m_scrollY += yOffset;
            }
        }

        Window::~Window()
        {
            if (m_pHandle != nullptr)
            {
                glfwDestroyWindow(m_pHandle);
            }
            Internal::g_windowRefCount--;
            if (Internal::g_windowRefCount == 0)
            {
                glfwTerminate();
            }
        }

        bool Window::ShouldClose() const
        {
            return glfwWindowShouldClose(m_pHandle) != 0;
        }

        void Window::PollEvents()
        {
            glfwPollEvents();
        }

        void Window::FramebufferSize(int32_t &outWidth,
                                      int32_t &outHeight) const
        {
            glfwGetFramebufferSize(m_pHandle, &outWidth, &outHeight);
        }

        void Window::CursorPos(double &outX,
                               double &outY) const
        {
            glfwGetCursorPos(m_pHandle, &outX, &outY);
        }

        bool Window::MouseButtonDown(MouseButton button) const
        {
            int32_t glfwButton = GLFW_MOUSE_BUTTON_LEFT;
            if (button == MouseButton::Right)
            {
                glfwButton = GLFW_MOUSE_BUTTON_RIGHT;
            }
            else if (button == MouseButton::Middle)
            {
                glfwButton = GLFW_MOUSE_BUTTON_MIDDLE;
            }
            return glfwGetMouseButton(m_pHandle, glfwButton) == GLFW_PRESS;
        }

        double Window::ConsumeScrollY()
        {
            double scrollY = m_scrollY;
            m_scrollY = 0.0;
            return scrollY;
        }

        bool Window::ConsumeFramebufferResized()
        {
            bool resized = m_framebufferResized;
            m_framebufferResized = false;
            return resized;
        }

        GLFWwindow *Window::Handle() const
        {
            return m_pHandle;
        }

        std::vector<const char *> Window::RequiredInstanceExtensions()
        {
            uint32_t count = 0;
            const char **ppExtensions = glfwGetRequiredInstanceExtensions(&count);
            if (ppExtensions == nullptr)
            {
                Logger::Instance().Log(LogSeverity::Error, "glfwGetRequiredInstanceExtensions failed — no usable Vulkan-capable windowing system?");
                throw std::runtime_error("glfwGetRequiredInstanceExtensions failed");
            }
            return std::vector<const char *>(ppExtensions, ppExtensions + count);
        }
    }
}
