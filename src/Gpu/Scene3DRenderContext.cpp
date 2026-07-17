#include "Hal/Gpu/Scene3DRenderContext.h"

#include "Hal/Color.h"
#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Gpu/ShaderPaths.h"
#include "Hal/Log/Logger.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace Hal
{
    namespace Gpu
    {
        // Push-constant block shared by both pipelines (matches the
        // SceneConstants block in SphereLambert.vert/.frag and
        // Line3D.vert): the model-view-projection matrix plus the Lambert
        // light direction. 80 bytes, within Vulkan's 128-byte guaranteed
        // push-constant minimum.
        struct Scene3DPushConstants
        {
            Mat4  viewProjection;
            float lightDir[4];
        };

        // Unit-sphere tessellation (a stacks x slices UV sphere). 16x24
        // gives ~2300 triangles per sphere — visually smooth at HAL's
        // typical agent sizes while keeping the shared mesh tiny.
        const int32_t g_sphereMeshStacks = 16;
        const int32_t g_sphereMeshSlices = 24;

        // Initial per-frame geometry capacities; EnsureSphereCapacity/
        // EnsureLineCapacity grow past these on demand.
        const uint32_t g_initialSphereCapacity = 1024;
        const uint32_t g_initialLineVertexCapacity = 2048;

        Scene3DRenderContext::Scene3DRenderContext(Window &window,
                                                   double  xDim,
                                                   double  yDim,
                                                   double  zDim)
            : m_window(window),
              m_centerX(xDim / 2.0),
              m_centerY(yDim / 2.0),
              m_centerZ(zDim / 2.0),
              m_maxDim(std::max(xDim, std::max(yDim, zDim))),
              m_yaw(0.6),
              m_pitch(0.5),
              m_distance(m_maxDim * 2.4),
              m_clearColor{0.0f, 0.0f, 0.0f, 1.0f},
              m_dragging(false),
              m_lastMouseX(0.0),
              m_lastMouseY(0.0),
              m_instance(VK_NULL_HANDLE),
              m_surface(VK_NULL_HANDLE),
              m_physicalDevice(VK_NULL_HANDLE),
              m_device(VK_NULL_HANDLE),
              m_graphicsQueueFamilyIndex(0),
              m_presentQueueFamilyIndex(0),
              m_graphicsQueue(VK_NULL_HANDLE),
              m_presentQueue(VK_NULL_HANDLE),
              m_swapchain(VK_NULL_HANDLE),
              m_swapchainFormat(VK_FORMAT_UNDEFINED),
              m_swapchainExtent{0, 0},
              m_depthFormat(VK_FORMAT_UNDEFINED),
              m_depthImage(VK_NULL_HANDLE),
              m_depthMemory(VK_NULL_HANDLE),
              m_depthImageView(VK_NULL_HANDLE),
              m_renderPass(VK_NULL_HANDLE),
              m_meshVertexBuffer(VK_NULL_HANDLE),
              m_meshVertexMemory(VK_NULL_HANDLE),
              m_meshIndexBuffer(VK_NULL_HANDLE),
              m_meshIndexMemory(VK_NULL_HANDLE),
              m_meshIndexCount(0),
              m_instanceBuffer(VK_NULL_HANDLE),
              m_instanceMemory(VK_NULL_HANDLE),
              m_instanceCapacity(0),
              m_lineBuffer(VK_NULL_HANDLE),
              m_lineMemory(VK_NULL_HANDLE),
              m_lineCapacity(0),
              m_sphereVertModule(VK_NULL_HANDLE),
              m_sphereFragModule(VK_NULL_HANDLE),
              m_lineVertModule(VK_NULL_HANDLE),
              m_lineFragModule(VK_NULL_HANDLE),
              m_pipelineLayout(VK_NULL_HANDLE),
              m_spherePipeline(VK_NULL_HANDLE),
              m_linePipeline(VK_NULL_HANDLE),
              m_commandPool(VK_NULL_HANDLE),
              m_imageAvailableSemaphore(VK_NULL_HANDLE),
              m_renderFinishedSemaphore(VK_NULL_HANDLE),
              m_inFlightFence(VK_NULL_HANDLE),
              m_captureEnabled(false),
              m_captureBuffer(VK_NULL_HANDLE),
              m_captureMemory(VK_NULL_HANDLE),
              m_captureSize(0),
              m_captureWidth(0),
              m_captureHeight(0)
        {
            // volkInitialize() and glfwInitVulkanLoader() already ran in
            // Window's constructor (they must precede glfwInit(), which
            // Window owns — see Window.cpp).
            CreateInstance();
            volkLoadInstance(m_instance);
            CreateSurface();
            SelectPhysicalDevice();
            CreateLogicalDevice();
            volkLoadDevice(m_device);
            vkGetDeviceQueue(m_device, m_graphicsQueueFamilyIndex, 0, &m_graphicsQueue);
            vkGetDeviceQueue(m_device, m_presentQueueFamilyIndex, 0, &m_presentQueue);
            CreateSwapchain();
            CreateDepthResources();
            CreateRenderPass();
            CreateFramebuffers();
            CreateSphereMesh();
            EnsureSphereCapacity(g_initialSphereCapacity);
            EnsureLineCapacity(g_initialLineVertexCapacity);
            CreatePipelines();
            CreateCommandObjects();
            CreateSyncObjects();
            Logger::Instance().Log(LogSeverity::Info, "Hal::Gpu::Scene3DRenderContext initialized");
        }

        Scene3DRenderContext::~Scene3DRenderContext()
        {
            if (m_device != VK_NULL_HANDLE)
            {
                vkDeviceWaitIdle(m_device);
            }
            DestroyCaptureBuffer();
            if (m_inFlightFence != VK_NULL_HANDLE)
            {
                vkDestroyFence(m_device, m_inFlightFence, nullptr);
            }
            if (m_renderFinishedSemaphore != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(m_device, m_renderFinishedSemaphore, nullptr);
            }
            if (m_imageAvailableSemaphore != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(m_device, m_imageAvailableSemaphore, nullptr);
            }
            if (m_commandPool != VK_NULL_HANDLE)
            {
                vkDestroyCommandPool(m_device, m_commandPool, nullptr);
            }
            if (m_spherePipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(m_device, m_spherePipeline, nullptr);
            }
            if (m_linePipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(m_device, m_linePipeline, nullptr);
            }
            if (m_pipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
            }
            if (m_sphereVertModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(m_device, m_sphereVertModule, nullptr);
            }
            if (m_sphereFragModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(m_device, m_sphereFragModule, nullptr);
            }
            if (m_lineVertModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(m_device, m_lineVertModule, nullptr);
            }
            if (m_lineFragModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(m_device, m_lineFragModule, nullptr);
            }
            if (m_lineBuffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(m_device, m_lineBuffer, nullptr);
            }
            if (m_lineMemory != VK_NULL_HANDLE)
            {
                vkFreeMemory(m_device, m_lineMemory, nullptr);
            }
            if (m_instanceBuffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(m_device, m_instanceBuffer, nullptr);
            }
            if (m_instanceMemory != VK_NULL_HANDLE)
            {
                vkFreeMemory(m_device, m_instanceMemory, nullptr);
            }
            if (m_meshIndexBuffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(m_device, m_meshIndexBuffer, nullptr);
            }
            if (m_meshIndexMemory != VK_NULL_HANDLE)
            {
                vkFreeMemory(m_device, m_meshIndexMemory, nullptr);
            }
            if (m_meshVertexBuffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(m_device, m_meshVertexBuffer, nullptr);
            }
            if (m_meshVertexMemory != VK_NULL_HANDLE)
            {
                vkFreeMemory(m_device, m_meshVertexMemory, nullptr);
            }
            for (VkFramebuffer framebuffer : m_framebuffers)
            {
                vkDestroyFramebuffer(m_device, framebuffer, nullptr);
            }
            if (m_renderPass != VK_NULL_HANDLE)
            {
                vkDestroyRenderPass(m_device, m_renderPass, nullptr);
            }
            if (m_depthImageView != VK_NULL_HANDLE)
            {
                vkDestroyImageView(m_device, m_depthImageView, nullptr);
            }
            if (m_depthImage != VK_NULL_HANDLE)
            {
                vkDestroyImage(m_device, m_depthImage, nullptr);
            }
            if (m_depthMemory != VK_NULL_HANDLE)
            {
                vkFreeMemory(m_device, m_depthMemory, nullptr);
            }
            for (VkImageView imageView : m_swapchainImageViews)
            {
                vkDestroyImageView(m_device, imageView, nullptr);
            }
            if (m_swapchain != VK_NULL_HANDLE)
            {
                vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
            }
            if (m_device != VK_NULL_HANDLE)
            {
                vkDestroyDevice(m_device, nullptr);
            }
            if (m_surface != VK_NULL_HANDLE)
            {
                vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
            }
            if (m_instance != VK_NULL_HANDLE)
            {
                vkDestroyInstance(m_instance, nullptr);
            }
        }

        void Scene3DRenderContext::SetCamera(double yaw,
                                             double pitch,
                                             double distance)
        {
            m_yaw = yaw;
            m_pitch = pitch;
            m_distance = distance;
        }

        void Scene3DRenderContext::HandleInput()
        {
            bool   down = m_window.MouseButtonDown(MouseButton::Left);
            double mouseX = 0.0;
            double mouseY = 0.0;
            m_window.CursorPos(mouseX, mouseY);
            if (down)
            {
                if (m_dragging)
                {
                    // Same sensitivity/clamp as the software Scene3D, so
                    // the two renderers feel identical to drive.
                    m_yaw += (mouseX - m_lastMouseX) * 0.01;
                    m_pitch += (mouseY - m_lastMouseY) * 0.01;
                    m_pitch = std::max(-1.5, std::min(1.5, m_pitch));
                }
                m_dragging = true;
                m_lastMouseX = mouseX;
                m_lastMouseY = mouseY;
            }
            else
            {
                m_dragging = false;
            }

            double scrollY = m_window.ConsumeScrollY();
            if (scrollY != 0.0)
            {
                m_distance *= std::pow(0.92, scrollY);
                m_distance = std::max(m_maxDim * 0.05, m_distance);
            }
        }

        void Scene3DRenderContext::Clear(int32_t bgColor)
        {
            m_clearColor[0] = static_cast<float>(GetRed(bgColor));
            m_clearColor[1] = static_cast<float>(GetGreen(bgColor));
            m_clearColor[2] = static_cast<float>(GetBlue(bgColor));
            m_clearColor[3] = 1.0f;
        }

        void Scene3DRenderContext::ClearBox(int32_t bgColor,
                                            int32_t lineColor)
        {
            Clear(bgColor);

            double x1 = m_centerX * 2.0;
            double y1 = m_centerY * 2.0;
            double z1 = m_centerZ * 2.0;

            // 4 edges along x, 4 along y, 4 along z.
            Line(0.0, 0.0, 0.0, x1, 0.0, 0.0, lineColor);
            Line(0.0, y1, 0.0, x1, y1, 0.0, lineColor);
            Line(0.0, 0.0, z1, x1, 0.0, z1, lineColor);
            Line(0.0, y1, z1, x1, y1, z1, lineColor);

            Line(0.0, 0.0, 0.0, 0.0, y1, 0.0, lineColor);
            Line(x1, 0.0, 0.0, x1, y1, 0.0, lineColor);
            Line(0.0, 0.0, z1, 0.0, y1, z1, lineColor);
            Line(x1, 0.0, z1, x1, y1, z1, lineColor);

            Line(0.0, 0.0, 0.0, 0.0, 0.0, z1, lineColor);
            Line(x1, 0.0, 0.0, x1, 0.0, z1, lineColor);
            Line(0.0, y1, 0.0, 0.0, y1, z1, lineColor);
            Line(x1, y1, 0.0, x1, y1, z1, lineColor);
        }

        void Scene3DRenderContext::Sphere(double  x,
                                          double  y,
                                          double  z,
                                          double  radius,
                                          int32_t color)
        {
            SphereInstance instance;
            instance.centerX = static_cast<float>(x);
            instance.centerY = static_cast<float>(y);
            instance.centerZ = static_cast<float>(z);
            instance.radius = static_cast<float>(radius);
            instance.r = static_cast<float>(GetRed(color));
            instance.g = static_cast<float>(GetGreen(color));
            instance.b = static_cast<float>(GetBlue(color));
            instance.a = 1.0f;
            m_sphereQueue.push_back(instance);
        }

        void Scene3DRenderContext::Line(double  x1,
                                        double  y1,
                                        double  z1,
                                        double  x2,
                                        double  y2,
                                        double  z2,
                                        int32_t color)
        {
            LineVertex vertex;
            vertex.r = static_cast<float>(GetRed(color));
            vertex.g = static_cast<float>(GetGreen(color));
            vertex.b = static_cast<float>(GetBlue(color));
            vertex.a = 1.0f;

            vertex.x = static_cast<float>(x1);
            vertex.y = static_cast<float>(y1);
            vertex.z = static_cast<float>(z1);
            m_lineQueue.push_back(vertex);

            vertex.x = static_cast<float>(x2);
            vertex.y = static_cast<float>(y2);
            vertex.z = static_cast<float>(z2);
            m_lineQueue.push_back(vertex);
        }

        void Scene3DRenderContext::CreateInstance()
        {
            uint32_t availableCount = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &availableCount, nullptr);
            std::vector<VkExtensionProperties> available(availableCount);
            vkEnumerateInstanceExtensionProperties(nullptr, &availableCount, available.data());

            std::vector<const char *> enabledExtensions = Window::RequiredInstanceExtensions();
            VkInstanceCreateFlags createFlags = 0;
            if (Internal::HasInstanceExtension(available, "VK_KHR_portability_enumeration"))
            {
                enabledExtensions.push_back("VK_KHR_portability_enumeration");
                createFlags |= 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
            }

            VkApplicationInfo appInfo = {};
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pApplicationName = "HalCpp";
            appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
            appInfo.pEngineName = "HalCpp";
            appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
            appInfo.apiVersion = VK_API_VERSION_1_0;

            VkInstanceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &appInfo;
            createInfo.flags = createFlags;
            createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
            createInfo.ppEnabledExtensionNames = enabledExtensions.empty() ? nullptr : enabledExtensions.data();

            Internal::CheckResult(vkCreateInstance(&createInfo, nullptr, &m_instance), "vkCreateInstance failed");
        }

        void Scene3DRenderContext::CreateSurface()
        {
            Internal::CheckResult(glfwCreateWindowSurface(m_instance, m_window.Handle(), nullptr, &m_surface), "glfwCreateWindowSurface failed");
        }

        void Scene3DRenderContext::SelectPhysicalDevice()
        {
            uint32_t deviceCount = 0;
            Internal::CheckResult(vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices failed");
            if (deviceCount == 0)
            {
                Logger::Instance().Log(LogSeverity::Error, "No Vulkan-capable physical devices found");
                throw std::runtime_error("no Vulkan-capable physical devices found");
            }

            std::vector<VkPhysicalDevice> devices(deviceCount);
            Internal::CheckResult(vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices failed");

            for (VkPhysicalDevice candidate : devices)
            {
                uint32_t queueFamilyCount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, nullptr);
                std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
                vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, queueFamilies.data());

                int32_t graphicsFamily = -1;
                int32_t presentFamily = -1;
                for (uint32_t i = 0; i < queueFamilyCount; i++)
                {
                    if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
                    {
                        graphicsFamily = static_cast<int32_t>(i);
                    }
                    VkBool32 presentSupport = VK_FALSE;
                    vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, m_surface, &presentSupport);
                    if (presentSupport == VK_TRUE)
                    {
                        presentFamily = static_cast<int32_t>(i);
                    }
                }

                if ((graphicsFamily >= 0) && (presentFamily >= 0))
                {
                    m_physicalDevice = candidate;
                    m_graphicsQueueFamilyIndex = static_cast<uint32_t>(graphicsFamily);
                    m_presentQueueFamilyIndex = static_cast<uint32_t>(presentFamily);
                    return;
                }
            }

            Logger::Instance().Log(LogSeverity::Error, "No physical device with both graphics and present queue families found");
            throw std::runtime_error("no physical device with both graphics and present queue families found");
        }

        void Scene3DRenderContext::CreateLogicalDevice()
        {
            std::vector<uint32_t> uniqueFamilies = {m_graphicsQueueFamilyIndex};
            if (m_presentQueueFamilyIndex != m_graphicsQueueFamilyIndex)
            {
                uniqueFamilies.push_back(m_presentQueueFamilyIndex);
            }

            const float queuePriority = 1.0f;
            std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
            for (uint32_t family : uniqueFamilies)
            {
                VkDeviceQueueCreateInfo queueCreateInfo = {};
                queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfo.queueFamilyIndex = family;
                queueCreateInfo.queueCount = 1;
                queueCreateInfo.pQueuePriorities = &queuePriority;
                queueCreateInfos.push_back(queueCreateInfo);
            }

            uint32_t availableCount = 0;
            vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &availableCount, nullptr);
            std::vector<VkExtensionProperties> available(availableCount);
            vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &availableCount, available.data());

            std::vector<const char *> enabledExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
            if (Internal::HasDeviceExtension(available, "VK_KHR_portability_subset"))
            {
                enabledExtensions.push_back("VK_KHR_portability_subset");
            }

            VkDeviceCreateInfo deviceCreateInfo = {};
            deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
            deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
            deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
            deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();

            Internal::CheckResult(vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device), "vkCreateDevice failed");
        }

        void Scene3DRenderContext::CreateSwapchain()
        {
            VkSurfaceCapabilitiesKHR capabilities;
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &capabilities);

            uint32_t formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, nullptr);
            std::vector<VkSurfaceFormatKHR> formats(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, formats.data());

            VkSurfaceFormatKHR chosenFormat = formats[0];
            for (const VkSurfaceFormatKHR &candidate : formats)
            {
                if ((candidate.format == VK_FORMAT_B8G8R8A8_SRGB) && (candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR))
                {
                    chosenFormat = candidate;
                    break;
                }
            }

            int32_t framebufferWidth = 0;
            int32_t framebufferHeight = 0;
            m_window.FramebufferSize(framebufferWidth, framebufferHeight);
            VkExtent2D extent;
            extent.width = std::clamp(static_cast<uint32_t>(framebufferWidth), capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            extent.height = std::clamp(static_cast<uint32_t>(framebufferHeight), capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            uint32_t imageCount = capabilities.minImageCount + 1;
            if ((capabilities.maxImageCount > 0) && (imageCount > capabilities.maxImageCount))
            {
                imageCount = capabilities.maxImageCount;
            }

            VkSwapchainCreateInfoKHR createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            createInfo.surface = m_surface;
            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = chosenFormat.format;
            createInfo.imageColorSpace = chosenFormat.colorSpace;
            createInfo.imageExtent = extent;
            createInfo.imageArrayLayers = 1;
            // TRANSFER_SRC lets Render() copy the presented image back to a
            // host buffer for frame capture. It is a creation-time flag
            // (can't be toggled per frame) but costs nothing at render time,
            // so it is always set; the actual readback is still gated behind
            // SetCaptureEnabled.
            createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

            uint32_t queueFamilyIndices[] = {m_graphicsQueueFamilyIndex, m_presentQueueFamilyIndex};
            if (m_graphicsQueueFamilyIndex != m_presentQueueFamilyIndex)
            {
                createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
                createInfo.queueFamilyIndexCount = 2;
                createInfo.pQueueFamilyIndices = queueFamilyIndices;
            }
            else
            {
                createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            }

            createInfo.preTransform = capabilities.currentTransform;
            createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
            createInfo.clipped = VK_TRUE;

            Internal::CheckResult(vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain), "vkCreateSwapchainKHR failed");

            m_swapchainFormat = chosenFormat.format;
            m_swapchainExtent = extent;

            uint32_t actualImageCount = 0;
            vkGetSwapchainImagesKHR(m_device, m_swapchain, &actualImageCount, nullptr);
            m_swapchainImages.resize(actualImageCount);
            vkGetSwapchainImagesKHR(m_device, m_swapchain, &actualImageCount, m_swapchainImages.data());

            m_swapchainImageViews.resize(actualImageCount);
            for (uint32_t i = 0; i < actualImageCount; i++)
            {
                VkImageViewCreateInfo viewInfo = {};
                viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewInfo.image = m_swapchainImages[i];
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = m_swapchainFormat;
                viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                viewInfo.subresourceRange.levelCount = 1;
                viewInfo.subresourceRange.layerCount = 1;
                Internal::CheckResult(vkCreateImageView(m_device, &viewInfo, nullptr, &m_swapchainImageViews[i]), "vkCreateImageView failed");
            }
        }

        void Scene3DRenderContext::CreateDepthResources()
        {
            // First depth-capable format the device supports with optimal
            // tiling; D32_SFLOAT leads since MoltenVK and desktop drivers
            // support it universally.
            const VkFormat candidates[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
            for (VkFormat candidate : candidates)
            {
                VkFormatProperties props;
                vkGetPhysicalDeviceFormatProperties(m_physicalDevice, candidate, &props);
                if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0)
                {
                    m_depthFormat = candidate;
                    break;
                }
            }
            if (m_depthFormat == VK_FORMAT_UNDEFINED)
            {
                Logger::Instance().Log(LogSeverity::Error, "No supported depth attachment format found");
                throw std::runtime_error("no supported depth attachment format found");
            }

            VkImageCreateInfo imageInfo = {};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent.width = m_swapchainExtent.width;
            imageInfo.extent.height = m_swapchainExtent.height;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = m_depthFormat;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            Internal::CheckResult(vkCreateImage(m_device, &imageInfo, nullptr, &m_depthImage), "vkCreateImage (depth) failed");

            VkMemoryRequirements memRequirements;
            vkGetImageMemoryRequirements(m_device, m_depthImage, &memRequirements);

            VkMemoryAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = Internal::FindMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Internal::CheckResult(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_depthMemory), "vkAllocateMemory (depth) failed");
            Internal::CheckResult(vkBindImageMemory(m_device, m_depthImage, m_depthMemory, 0), "vkBindImageMemory (depth) failed");

            VkImageViewCreateInfo viewInfo = {};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_depthImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = m_depthFormat;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            Internal::CheckResult(vkCreateImageView(m_device, &viewInfo, nullptr, &m_depthImageView), "vkCreateImageView (depth) failed");
        }

        void Scene3DRenderContext::CreateRenderPass()
        {
            VkAttachmentDescription colorAttachment = {};
            colorAttachment.format = m_swapchainFormat;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            // The depth buffer only matters within the frame, so its
            // contents are cleared on load and discarded on store.
            VkAttachmentDescription depthAttachment = {};
            depthAttachment.format = m_depthFormat;
            depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentReference colorAttachmentRef = {};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentReference depthAttachmentRef = {};
            depthAttachmentRef.attachment = 1;
            depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass = {};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;
            subpass.pDepthStencilAttachment = &depthAttachmentRef;

            VkSubpassDependency dependency = {};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency.srcAccessMask = 0;
            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            VkAttachmentDescription attachments[] = {colorAttachment, depthAttachment};
            VkRenderPassCreateInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = 2;
            renderPassInfo.pAttachments = attachments;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = 1;
            renderPassInfo.pDependencies = &dependency;

            Internal::CheckResult(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass), "vkCreateRenderPass failed");
        }

        void Scene3DRenderContext::CreateFramebuffers()
        {
            m_framebuffers.resize(m_swapchainImageViews.size());
            for (size_t i = 0; i < m_swapchainImageViews.size(); i++)
            {
                // The single depth image is shared across framebuffers —
                // safe with one frame in flight.
                VkImageView attachments[] = {m_swapchainImageViews[i], m_depthImageView};
                VkFramebufferCreateInfo framebufferInfo = {};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = m_renderPass;
                framebufferInfo.attachmentCount = 2;
                framebufferInfo.pAttachments = attachments;
                framebufferInfo.width = m_swapchainExtent.width;
                framebufferInfo.height = m_swapchainExtent.height;
                framebufferInfo.layers = 1;
                Internal::CheckResult(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_framebuffers[i]), "vkCreateFramebuffer failed");
            }
        }

        void Scene3DRenderContext::CreateSphereMesh()
        {
            // stacks x slices UV sphere on the unit radius. The seam
            // column and both poles are duplicated per ring so indexing
            // stays a simple quad grid.
            std::vector<SphereMeshVertex> vertices;
            vertices.reserve(static_cast<size_t>((g_sphereMeshStacks + 1) * (g_sphereMeshSlices + 1)));
            for (int32_t stack = 0; stack <= g_sphereMeshStacks; stack++)
            {
                double phi = 3.14159265358979323846 * static_cast<double>(stack) / g_sphereMeshStacks;
                for (int32_t slice = 0; slice <= g_sphereMeshSlices; slice++)
                {
                    double theta = 2.0 * 3.14159265358979323846 * static_cast<double>(slice) / g_sphereMeshSlices;
                    SphereMeshVertex vertex;
                    vertex.x = static_cast<float>(std::sin(phi) * std::cos(theta));
                    vertex.y = static_cast<float>(std::cos(phi));
                    vertex.z = static_cast<float>(std::sin(phi) * std::sin(theta));
                    vertices.push_back(vertex);
                }
            }

            std::vector<uint32_t> indices;
            indices.reserve(static_cast<size_t>(g_sphereMeshStacks * g_sphereMeshSlices * 6));
            for (int32_t stack = 0; stack < g_sphereMeshStacks; stack++)
            {
                for (int32_t slice = 0; slice < g_sphereMeshSlices; slice++)
                {
                    uint32_t topLeft = static_cast<uint32_t>(stack * (g_sphereMeshSlices + 1) + slice);
                    uint32_t bottomLeft = topLeft + static_cast<uint32_t>(g_sphereMeshSlices + 1);
                    indices.push_back(topLeft);
                    indices.push_back(bottomLeft);
                    indices.push_back(topLeft + 1);
                    indices.push_back(topLeft + 1);
                    indices.push_back(bottomLeft);
                    indices.push_back(bottomLeft + 1);
                }
            }
            m_meshIndexCount = static_cast<uint32_t>(indices.size());

            VkDeviceSize vertexSize = sizeof(SphereMeshVertex) * vertices.size();
            m_meshVertexBuffer = Internal::CreateBuffer(
                m_physicalDevice,
                m_device,
                vertexSize,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_meshVertexMemory);

            void *pMapped = nullptr;
            Internal::CheckResult(vkMapMemory(m_device, m_meshVertexMemory, 0, vertexSize, 0, &pMapped), "vkMapMemory (sphere mesh vertices) failed");
            std::memcpy(pMapped, vertices.data(), static_cast<size_t>(vertexSize));
            vkUnmapMemory(m_device, m_meshVertexMemory);

            VkDeviceSize indexSize = sizeof(uint32_t) * indices.size();
            m_meshIndexBuffer = Internal::CreateBuffer(
                m_physicalDevice,
                m_device,
                indexSize,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_meshIndexMemory);

            Internal::CheckResult(vkMapMemory(m_device, m_meshIndexMemory, 0, indexSize, 0, &pMapped), "vkMapMemory (sphere mesh indices) failed");
            std::memcpy(pMapped, indices.data(), static_cast<size_t>(indexSize));
            vkUnmapMemory(m_device, m_meshIndexMemory);
        }

        void Scene3DRenderContext::EnsureSphereCapacity(uint32_t instanceCount)
        {
            if ((instanceCount <= m_instanceCapacity) || (instanceCount == 0))
            {
                return;
            }
            if (m_instanceBuffer != VK_NULL_HANDLE)
            {
                vkDeviceWaitIdle(m_device);
                vkDestroyBuffer(m_device, m_instanceBuffer, nullptr);
                vkFreeMemory(m_device, m_instanceMemory, nullptr);
            }
            uint32_t newCapacity = std::max(instanceCount, m_instanceCapacity * 2);
            m_instanceBuffer = Internal::CreateBuffer(
                m_physicalDevice,
                m_device,
                sizeof(SphereInstance) * static_cast<VkDeviceSize>(newCapacity),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_instanceMemory);
            m_instanceCapacity = newCapacity;
        }

        void Scene3DRenderContext::EnsureLineCapacity(uint32_t vertexCount)
        {
            if ((vertexCount <= m_lineCapacity) || (vertexCount == 0))
            {
                return;
            }
            if (m_lineBuffer != VK_NULL_HANDLE)
            {
                vkDeviceWaitIdle(m_device);
                vkDestroyBuffer(m_device, m_lineBuffer, nullptr);
                vkFreeMemory(m_device, m_lineMemory, nullptr);
            }
            uint32_t newCapacity = std::max(vertexCount, m_lineCapacity * 2);
            m_lineBuffer = Internal::CreateBuffer(
                m_physicalDevice,
                m_device,
                sizeof(LineVertex) * static_cast<VkDeviceSize>(newCapacity),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_lineMemory);
            m_lineCapacity = newCapacity;
        }

        VkShaderModule Scene3DRenderContext::CreateShaderModule(const char *pShaderName)
        {
            std::vector<char> code = Internal::ReadFile(ShaderPath(pShaderName));
            VkShaderModuleCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
            VkShaderModule shaderModule = VK_NULL_HANDLE;
            Internal::CheckResult(vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule), "vkCreateShaderModule failed");
            return shaderModule;
        }

        void Scene3DRenderContext::CreatePipelines()
        {
            m_sphereVertModule = CreateShaderModule("SphereLambert.vert");
            m_sphereFragModule = CreateShaderModule("SphereLambert.frag");
            m_lineVertModule = CreateShaderModule("Line3D.vert");
            m_lineFragModule = CreateShaderModule("Line3D.frag");

            // One layout for both pipelines: the shared push-constant
            // block (mvp + lightDir), no descriptor sets. SphereLambert
            // reads lightDir in its fragment stage, so the range spans
            // both stages.
            VkPushConstantRange pushRange = {};
            pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            pushRange.offset = 0;
            pushRange.size = sizeof(Scene3DPushConstants);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            Internal::CheckResult(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout), "vkCreatePipelineLayout failed");

            // State shared by both pipelines. Viewport and scissor are
            // dynamic (recorded per frame from the current swapchain
            // extent) so a window resize only recreates the
            // swapchain-sized resources, never these pipelines.
            VkPipelineViewportStateCreateInfo viewportState = {};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamicState = {};
            dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicState.dynamicStateCount = 2;
            dynamicState.pDynamicStates = dynamicStates;

            VkPipelineRasterizationStateCreateInfo rasterizer = {};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.cullMode = VK_CULL_MODE_NONE;
            rasterizer.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisampling = {};
            multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depthStencil = {};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

            VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo colorBlending = {};
            colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;

            // Sphere pipeline: binding 0 walks the shared unit-sphere
            // mesh per vertex, binding 1 advances per instance.
            VkVertexInputBindingDescription sphereBindings[2] = {};
            sphereBindings[0].binding = 0;
            sphereBindings[0].stride = sizeof(SphereMeshVertex);
            sphereBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            sphereBindings[1].binding = 1;
            sphereBindings[1].stride = sizeof(SphereInstance);
            sphereBindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

            VkVertexInputAttributeDescription sphereAttributes[3] = {};
            sphereAttributes[0].location = 0;
            sphereAttributes[0].binding = 0;
            sphereAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            sphereAttributes[0].offset = 0;
            sphereAttributes[1].location = 1;
            sphereAttributes[1].binding = 1;
            sphereAttributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            sphereAttributes[1].offset = 0;
            sphereAttributes[2].location = 2;
            sphereAttributes[2].binding = 1;
            sphereAttributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            sphereAttributes[2].offset = sizeof(float) * 4;

            VkPipelineVertexInputStateCreateInfo sphereVertexInput = {};
            sphereVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            sphereVertexInput.vertexBindingDescriptionCount = 2;
            sphereVertexInput.pVertexBindingDescriptions = sphereBindings;
            sphereVertexInput.vertexAttributeDescriptionCount = 3;
            sphereVertexInput.pVertexAttributeDescriptions = sphereAttributes;

            VkPipelineInputAssemblyStateCreateInfo triangleAssembly = {};
            triangleAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            triangleAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineShaderStageCreateInfo sphereStages[2] = {};
            sphereStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            sphereStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            sphereStages[0].module = m_sphereVertModule;
            sphereStages[0].pName = "main";
            sphereStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            sphereStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            sphereStages[1].module = m_sphereFragModule;
            sphereStages[1].pName = "main";

            VkGraphicsPipelineCreateInfo spherePipelineInfo = {};
            spherePipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            spherePipelineInfo.stageCount = 2;
            spherePipelineInfo.pStages = sphereStages;
            spherePipelineInfo.pVertexInputState = &sphereVertexInput;
            spherePipelineInfo.pInputAssemblyState = &triangleAssembly;
            spherePipelineInfo.pViewportState = &viewportState;
            spherePipelineInfo.pRasterizationState = &rasterizer;
            spherePipelineInfo.pMultisampleState = &multisampling;
            spherePipelineInfo.pDepthStencilState = &depthStencil;
            spherePipelineInfo.pColorBlendState = &colorBlending;
            spherePipelineInfo.pDynamicState = &dynamicState;
            spherePipelineInfo.layout = m_pipelineLayout;
            spherePipelineInfo.renderPass = m_renderPass;
            spherePipelineInfo.subpass = 0;

            Internal::CheckResult(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &spherePipelineInfo, nullptr, &m_spherePipeline), "vkCreateGraphicsPipelines (spheres) failed");

            // Line pipeline: one per-vertex binding, line-list topology,
            // same depth/blend/layout state.
            VkVertexInputBindingDescription lineBinding = {};
            lineBinding.binding = 0;
            lineBinding.stride = sizeof(LineVertex);
            lineBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription lineAttributes[2] = {};
            lineAttributes[0].location = 0;
            lineAttributes[0].binding = 0;
            lineAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            lineAttributes[0].offset = 0;
            lineAttributes[1].location = 1;
            lineAttributes[1].binding = 0;
            lineAttributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            lineAttributes[1].offset = sizeof(float) * 3;

            VkPipelineVertexInputStateCreateInfo lineVertexInput = {};
            lineVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            lineVertexInput.vertexBindingDescriptionCount = 1;
            lineVertexInput.pVertexBindingDescriptions = &lineBinding;
            lineVertexInput.vertexAttributeDescriptionCount = 2;
            lineVertexInput.pVertexAttributeDescriptions = lineAttributes;

            VkPipelineInputAssemblyStateCreateInfo lineAssembly = {};
            lineAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            lineAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

            VkPipelineShaderStageCreateInfo lineStages[2] = {};
            lineStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            lineStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            lineStages[0].module = m_lineVertModule;
            lineStages[0].pName = "main";
            lineStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            lineStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            lineStages[1].module = m_lineFragModule;
            lineStages[1].pName = "main";

            VkGraphicsPipelineCreateInfo linePipelineInfo = {};
            linePipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            linePipelineInfo.stageCount = 2;
            linePipelineInfo.pStages = lineStages;
            linePipelineInfo.pVertexInputState = &lineVertexInput;
            linePipelineInfo.pInputAssemblyState = &lineAssembly;
            linePipelineInfo.pViewportState = &viewportState;
            linePipelineInfo.pRasterizationState = &rasterizer;
            linePipelineInfo.pMultisampleState = &multisampling;
            linePipelineInfo.pDepthStencilState = &depthStencil;
            linePipelineInfo.pColorBlendState = &colorBlending;
            linePipelineInfo.pDynamicState = &dynamicState;
            linePipelineInfo.layout = m_pipelineLayout;
            linePipelineInfo.renderPass = m_renderPass;
            linePipelineInfo.subpass = 0;

            Internal::CheckResult(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &linePipelineInfo, nullptr, &m_linePipeline), "vkCreateGraphicsPipelines (lines) failed");
        }

        void Scene3DRenderContext::CreateCommandObjects()
        {
            VkCommandPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
            Internal::CheckResult(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool), "vkCreateCommandPool failed");

            m_commandBuffers.resize(1);
            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = m_commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;
            Internal::CheckResult(vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()), "vkAllocateCommandBuffers failed");
        }

        void Scene3DRenderContext::CreateSyncObjects()
        {
            VkSemaphoreCreateInfo semaphoreInfo = {};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            Internal::CheckResult(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphore), "vkCreateSemaphore failed");
            Internal::CheckResult(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphore), "vkCreateSemaphore failed");

            VkFenceCreateInfo fenceInfo = {};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            Internal::CheckResult(vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFence), "vkCreateFence failed");
        }

        void Scene3DRenderContext::ComputeViewProjection(Mat4 &outViewProjection,
                                                         Vec3 &outLightDir) const
        {
            // Same orbit convention as the software Scene3D (yaw about
            // the world's vertical y axis, then pitch), expressed as a
            // camera position rather than point rotations: the camera
            // sits `distance` from the domain center along the inverse-
            // rotated view axis, looking back at the center with world
            // +y up.
            float cosYaw = static_cast<float>(std::cos(m_yaw));
            float sinYaw = static_cast<float>(std::sin(m_yaw));
            float cosPitch = static_cast<float>(std::cos(m_pitch));
            float sinPitch = static_cast<float>(std::sin(m_pitch));
            float dist = static_cast<float>(m_distance);

            Vec3 center = {static_cast<float>(m_centerX), static_cast<float>(m_centerY), static_cast<float>(m_centerZ)};
            Vec3 eye = {center.x - dist * cosPitch * sinYaw,
                        center.y - dist * sinPitch,
                        center.z - dist * cosPitch * cosYaw};

            Vec3 worldUp = {0.0f, 1.0f, 0.0f};
            Mat4 view = Mat4LookAt(eye, center, worldUp);

            // Vertical field of view matching the software renderer's
            // focal length (focal = 0.9 * window height ->
            // fovY = 2 * atan(0.5 / 0.9)), so both renderers frame the
            // domain identically.
            float fovY = 2.0f * std::atan(0.5f / 0.9f);
            float aspect = static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);
            float nearPlane = static_cast<float>(m_maxDim) * 0.01f;
            float farPlane = dist + static_cast<float>(m_maxDim) * 4.0f;
            Mat4 projection = Mat4Perspective(fovY, aspect, nearPlane, farPlane);

            outViewProjection = Mat4Multiply(projection, view);

            // Headlight: light arrives from the camera's direction, so
            // the sphere face toward the viewer is the lit one.
            outLightDir = Vec3Normalize(Vec3Sub(eye, center));
        }

        bool Scene3DRenderContext::RecreateSwapchain()
        {
            int32_t framebufferWidth = 0;
            int32_t framebufferHeight = 0;
            m_window.FramebufferSize(framebufferWidth, framebufferHeight);
            if ((framebufferWidth == 0) || (framebufferHeight == 0))
            {
                // Minimized: a zero-extent swapchain is invalid, so leave
                // the old (out-of-date) one in place and skip frames
                // until the window is restored.
                return false;
            }

            vkDeviceWaitIdle(m_device);

            for (VkFramebuffer framebuffer : m_framebuffers)
            {
                vkDestroyFramebuffer(m_device, framebuffer, nullptr);
            }
            m_framebuffers.clear();
            vkDestroyImageView(m_device, m_depthImageView, nullptr);
            m_depthImageView = VK_NULL_HANDLE;
            vkDestroyImage(m_device, m_depthImage, nullptr);
            m_depthImage = VK_NULL_HANDLE;
            vkFreeMemory(m_device, m_depthMemory, nullptr);
            m_depthMemory = VK_NULL_HANDLE;
            for (VkImageView imageView : m_swapchainImageViews)
            {
                vkDestroyImageView(m_device, imageView, nullptr);
            }
            m_swapchainImageViews.clear();
            vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
            m_swapchain = VK_NULL_HANDLE;

            CreateSwapchain();
            CreateDepthResources();
            CreateFramebuffers();
            if (m_captureEnabled)
            {
                // Match the capture buffer to the new swapchain extent.
                CreateCaptureBuffer();
            }
            Logger::Instance().Log(LogSeverity::Info, "Scene3DRenderContext: swapchain recreated at " + std::to_string(framebufferWidth) + "x" + std::to_string(framebufferHeight));
            return true;
        }

        void Scene3DRenderContext::CreateCaptureBuffer()
        {
            DestroyCaptureBuffer();

            m_captureWidth = m_swapchainExtent.width;
            m_captureHeight = m_swapchainExtent.height;
            m_captureSize = static_cast<VkDeviceSize>(m_captureWidth) * static_cast<VkDeviceSize>(m_captureHeight) * 4;
            if (m_captureSize == 0)
            {
                return;
            }
            m_captureBuffer = Internal::CreateBuffer(
                m_physicalDevice,
                m_device,
                m_captureSize,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_captureMemory);
        }

        void Scene3DRenderContext::DestroyCaptureBuffer()
        {
            if (m_captureBuffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(m_device, m_captureBuffer, nullptr);
                m_captureBuffer = VK_NULL_HANDLE;
            }
            if (m_captureMemory != VK_NULL_HANDLE)
            {
                vkFreeMemory(m_device, m_captureMemory, nullptr);
                m_captureMemory = VK_NULL_HANDLE;
            }
            m_captureSize = 0;
            m_captureWidth = 0;
            m_captureHeight = 0;
        }

        void Scene3DRenderContext::SetCaptureEnabled(bool enabled)
        {
            if (enabled == m_captureEnabled)
            {
                return;
            }
            m_captureEnabled = enabled;
            if (enabled)
            {
                CreateCaptureBuffer();
            }
            else
            {
                vkDeviceWaitIdle(m_device);
                DestroyCaptureBuffer();
            }
        }

        bool Scene3DRenderContext::ReadCapturedFrame(std::vector<uint8_t> &outRgba,
                                                     int32_t              &outWidth,
                                                     int32_t              &outHeight)
        {
            if (!m_captureEnabled || (m_captureBuffer == VK_NULL_HANDLE))
            {
                return false;
            }

            // The copy is part of the last submitted frame; wait for it.
            vkWaitForFences(m_device, 1, &m_inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());

            outWidth = static_cast<int32_t>(m_captureWidth);
            outHeight = static_cast<int32_t>(m_captureHeight);
            outRgba.resize(static_cast<size_t>(m_captureSize));

            void *pMapped = nullptr;
            Internal::CheckResult(vkMapMemory(m_device, m_captureMemory, 0, m_captureSize, 0, &pMapped), "vkMapMemory (capture) failed");

            const uint8_t *pSrc = static_cast<const uint8_t *>(pMapped);
            // The swapchain is almost always B8G8R8A8; swap R/B so the
            // output is RGBA regardless. R8G8B8A8 formats copy straight.
            bool swapRB = (m_swapchainFormat == VK_FORMAT_B8G8R8A8_UNORM) || (m_swapchainFormat == VK_FORMAT_B8G8R8A8_SRGB);
            size_t pixels = static_cast<size_t>(m_captureWidth) * static_cast<size_t>(m_captureHeight);
            for (size_t i = 0; i < pixels; i++)
            {
                size_t base = i * 4;
                if (swapRB)
                {
                    outRgba[base + 0] = pSrc[base + 2];
                    outRgba[base + 1] = pSrc[base + 1];
                    outRgba[base + 2] = pSrc[base + 0];
                    outRgba[base + 3] = pSrc[base + 3];
                }
                else
                {
                    outRgba[base + 0] = pSrc[base + 0];
                    outRgba[base + 1] = pSrc[base + 1];
                    outRgba[base + 2] = pSrc[base + 2];
                    outRgba[base + 3] = pSrc[base + 3];
                }
            }
            vkUnmapMemory(m_device, m_captureMemory);
            return true;
        }

        bool Scene3DRenderContext::Render()
        {
            uint32_t sphereCount = static_cast<uint32_t>(m_sphereQueue.size());
            uint32_t lineVertexCount = static_cast<uint32_t>(m_lineQueue.size());

            if (m_window.ConsumeFramebufferResized())
            {
                if (!RecreateSwapchain())
                {
                    m_sphereQueue.clear();
                    m_lineQueue.clear();
                    return false;
                }
            }

            EnsureSphereCapacity(sphereCount);
            EnsureLineCapacity(lineVertexCount);

            vkWaitForFences(m_device, 1, &m_inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());

            if (sphereCount > 0)
            {
                VkDeviceSize copySize = sizeof(SphereInstance) * static_cast<VkDeviceSize>(sphereCount);
                void *pMapped = nullptr;
                Internal::CheckResult(vkMapMemory(m_device, m_instanceMemory, 0, copySize, 0, &pMapped), "vkMapMemory (sphere instances) failed");
                std::memcpy(pMapped, m_sphereQueue.data(), static_cast<size_t>(copySize));
                vkUnmapMemory(m_device, m_instanceMemory);
            }
            if (lineVertexCount > 0)
            {
                VkDeviceSize copySize = sizeof(LineVertex) * static_cast<VkDeviceSize>(lineVertexCount);
                void *pMapped = nullptr;
                Internal::CheckResult(vkMapMemory(m_device, m_lineMemory, 0, copySize, 0, &pMapped), "vkMapMemory (line vertices) failed");
                std::memcpy(pMapped, m_lineQueue.data(), static_cast<size_t>(copySize));
                vkUnmapMemory(m_device, m_lineMemory);
            }

            // The frame's geometry is uploaded; the queues are consumed
            // whether or not the present below succeeds.
            m_sphereQueue.clear();
            m_lineQueue.clear();

            uint32_t imageIndex = 0;
            VkResult acquireResult = vkAcquireNextImageKHR(m_device, m_swapchain, std::numeric_limits<uint64_t>::max(), m_imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
            if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
            {
                // The acquire failed outright (nothing was signaled), so
                // recreate now and let the next Render draw. A SUBOPTIMAL
                // acquire, by contrast, DID succeed and signal the
                // semaphore, so that frame proceeds and present's result
                // triggers the recreation instead.
                RecreateSwapchain();
                return false;
            }
            if (acquireResult != VK_SUBOPTIMAL_KHR)
            {
                Internal::CheckResult(acquireResult, "vkAcquireNextImageKHR failed");
            }

            vkResetFences(m_device, 1, &m_inFlightFence);

            VkCommandBuffer commandBuffer = m_commandBuffers[0];
            Internal::CheckResult(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer failed");

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            Internal::CheckResult(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            VkClearValue clearValues[2] = {};
            clearValues[0].color = {{m_clearColor[0], m_clearColor[1], m_clearColor[2], m_clearColor[3]}};
            clearValues[1].depthStencil = {1.0f, 0};

            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = m_renderPass;
            renderPassInfo.framebuffer = m_framebuffers[imageIndex];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = m_swapchainExtent;
            renderPassInfo.clearValueCount = 2;
            renderPassInfo.pClearValues = clearValues;

            vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            // Both pipelines take viewport/scissor as dynamic state, so
            // one recording covers all the frame's draws.
            VkViewport viewport = {};
            viewport.width = static_cast<float>(m_swapchainExtent.width);
            viewport.height = static_cast<float>(m_swapchainExtent.height);
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor = {};
            scissor.extent = m_swapchainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            // One shared push-constant block serves both pipelines (same
            // layout), so it is recorded once per frame.
            Scene3DPushConstants pushConstants = {};
            Vec3 lightDir = {0.0f, 0.0f, 0.0f};
            ComputeViewProjection(pushConstants.viewProjection, lightDir);
            pushConstants.lightDir[0] = lightDir.x;
            pushConstants.lightDir[1] = lightDir.y;
            pushConstants.lightDir[2] = lightDir.z;
            pushConstants.lightDir[3] = 0.0f;
            vkCmdPushConstants(commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Scene3DPushConstants), &pushConstants);

            if (sphereCount > 0)
            {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_spherePipeline);
                VkBuffer     vertexBuffers[] = {m_meshVertexBuffer, m_instanceBuffer};
                VkDeviceSize offsets[] = {0, 0};
                vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
                vkCmdBindIndexBuffer(commandBuffer, m_meshIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(commandBuffer, m_meshIndexCount, sphereCount, 0, 0, 0);
            }

            if (lineVertexCount > 0)
            {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_linePipeline);
                VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, &m_lineBuffer, &offset);
                vkCmdDraw(commandBuffer, lineVertexCount, 1, 0, 0);
            }

            vkCmdEndRenderPass(commandBuffer);

            // Frame capture: the render pass left the image in
            // PRESENT_SRC_KHR; transition it to a transfer source, copy it
            // into the host-visible capture buffer, then transition it back
            // so it can still be presented below.
            if (m_captureEnabled &&
                (m_captureBuffer != VK_NULL_HANDLE) &&
                (m_captureWidth == m_swapchainExtent.width) &&
                (m_captureHeight == m_swapchainExtent.height))
            {
                VkImageMemoryBarrier toTransfer = {};
                toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                toTransfer.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toTransfer.image = m_swapchainImages[imageIndex];
                toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                toTransfer.subresourceRange.levelCount = 1;
                toTransfer.subresourceRange.layerCount = 1;
                toTransfer.srcAccessMask = 0;
                toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransfer);

                VkBufferImageCopy region = {};
                region.bufferOffset = 0;
                region.bufferRowLength = 0;
                region.bufferImageHeight = 0;
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0, 0, 0};
                region.imageExtent = {m_swapchainExtent.width, m_swapchainExtent.height, 1};
                vkCmdCopyImageToBuffer(commandBuffer, m_swapchainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_captureBuffer, 1, &region);

                VkImageMemoryBarrier toPresent = {};
                toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toPresent.image = m_swapchainImages[imageIndex];
                toPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                toPresent.subresourceRange.levelCount = 1;
                toPresent.subresourceRange.layerCount = 1;
                toPresent.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                toPresent.dstAccessMask = 0;
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &toPresent);
            }

            Internal::CheckResult(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer failed");

            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &m_imageAvailableSemaphore;
            submitInfo.pWaitDstStageMask = &waitStage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &m_renderFinishedSemaphore;

            Internal::CheckResult(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFence), "vkQueueSubmit failed");

            VkPresentInfoKHR presentInfo = {};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &m_renderFinishedSemaphore;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &m_swapchain;
            presentInfo.pImageIndices = &imageIndex;

            VkResult presentResult = vkQueuePresentKHR(m_presentQueue, &presentInfo);
            if ((presentResult == VK_ERROR_OUT_OF_DATE_KHR) || (presentResult == VK_SUBOPTIMAL_KHR))
            {
                // SUBOPTIMAL still presented this frame; OUT_OF_DATE did
                // not. Either way the swapchain is stale — rebuild it for
                // the next frame.
                RecreateSwapchain();
                return presentResult == VK_SUBOPTIMAL_KHR;
            }
            Internal::CheckResult(presentResult, "vkQueuePresentKHR failed");

            return true;
        }
    }
}
