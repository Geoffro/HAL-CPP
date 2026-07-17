#include "Hal/Gpu/WindowRenderContext.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Gpu/ShaderPaths.h"
#include "Hal/Log/Logger.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace Hal
{
    namespace Gpu
    {
        WindowRenderContext::WindowRenderContext(Window  &window,
                                                 int32_t  textureWidth,
                                                 int32_t  textureHeight)
            : m_window(window),
              m_textureWidth(textureWidth),
              m_textureHeight(textureHeight),
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
              m_renderPass(VK_NULL_HANDLE),
              m_textureImage(VK_NULL_HANDLE),
              m_textureMemory(VK_NULL_HANDLE),
              m_textureImageView(VK_NULL_HANDLE),
              m_textureSampler(VK_NULL_HANDLE),
              m_stagingBuffer(VK_NULL_HANDLE),
              m_stagingMemory(VK_NULL_HANDLE),
              m_descriptorSetLayout(VK_NULL_HANDLE),
              m_descriptorPool(VK_NULL_HANDLE),
              m_descriptorSet(VK_NULL_HANDLE),
              m_vertShaderModule(VK_NULL_HANDLE),
              m_fragShaderModule(VK_NULL_HANDLE),
              m_pipelineLayout(VK_NULL_HANDLE),
              m_pipeline(VK_NULL_HANDLE),
              m_commandPool(VK_NULL_HANDLE),
              m_imageAvailableSemaphore(VK_NULL_HANDLE),
              m_renderFinishedSemaphore(VK_NULL_HANDLE),
              m_inFlightFence(VK_NULL_HANDLE)
        {
            // volkInitialize() and glfwInitVulkanLoader() already happened
            // in Window's constructor (they must run before glfwInit(),
            // which Window owns — see Window.cpp) — volk's function
            // pointers are already populated by the time a Window exists
            // for this WindowRenderContext to be built against.
            CreateInstance();
            volkLoadInstance(m_instance);
            CreateSurface();
            SelectPhysicalDevice();
            CreateLogicalDevice();
            volkLoadDevice(m_device);
            vkGetDeviceQueue(m_device, m_graphicsQueueFamilyIndex, 0, &m_graphicsQueue);
            vkGetDeviceQueue(m_device, m_presentQueueFamilyIndex, 0, &m_presentQueue);
            CreateSwapchain();
            CreateRenderPass();
            CreateFramebuffers();
            CreateCommandObjects();
            CreateTextureResources();
            CreateGraphicsPipeline();
            CreateSyncObjects();
            Logger::Instance().Log(LogSeverity::Info, "Hal::Gpu::WindowRenderContext initialized");
        }

        WindowRenderContext::~WindowRenderContext()
        {
            if (m_device != VK_NULL_HANDLE)
            {
                vkDeviceWaitIdle(m_device);
            }
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
            if (m_pipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(m_device, m_pipeline, nullptr);
            }
            if (m_pipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
            }
            if (m_fragShaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(m_device, m_fragShaderModule, nullptr);
            }
            if (m_vertShaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(m_device, m_vertShaderModule, nullptr);
            }
            if (m_descriptorPool != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            }
            if (m_descriptorSetLayout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
            }
            if (m_stagingBuffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(m_device, m_stagingBuffer, nullptr);
            }
            if (m_stagingMemory != VK_NULL_HANDLE)
            {
                vkFreeMemory(m_device, m_stagingMemory, nullptr);
            }
            if (m_textureSampler != VK_NULL_HANDLE)
            {
                vkDestroySampler(m_device, m_textureSampler, nullptr);
            }
            if (m_textureImageView != VK_NULL_HANDLE)
            {
                vkDestroyImageView(m_device, m_textureImageView, nullptr);
            }
            if (m_textureImage != VK_NULL_HANDLE)
            {
                vkDestroyImage(m_device, m_textureImage, nullptr);
            }
            if (m_textureMemory != VK_NULL_HANDLE)
            {
                vkFreeMemory(m_device, m_textureMemory, nullptr);
            }
            if (m_commandPool != VK_NULL_HANDLE)
            {
                vkDestroyCommandPool(m_device, m_commandPool, nullptr);
            }
            for (VkFramebuffer framebuffer : m_framebuffers)
            {
                vkDestroyFramebuffer(m_device, framebuffer, nullptr);
            }
            if (m_renderPass != VK_NULL_HANDLE)
            {
                vkDestroyRenderPass(m_device, m_renderPass, nullptr);
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

        VkInstance WindowRenderContext::Instance() const
        {
            return m_instance;
        }

        VkPhysicalDevice WindowRenderContext::PhysicalDevice() const
        {
            return m_physicalDevice;
        }

        VkDevice WindowRenderContext::Device() const
        {
            return m_device;
        }

        uint32_t WindowRenderContext::GraphicsQueueFamilyIndex() const
        {
            return m_graphicsQueueFamilyIndex;
        }

        VkQueue WindowRenderContext::GraphicsQueue() const
        {
            return m_graphicsQueue;
        }

        VkRenderPass WindowRenderContext::RenderPass() const
        {
            return m_renderPass;
        }

        uint32_t WindowRenderContext::SwapchainImageCount() const
        {
            return static_cast<uint32_t>(m_swapchainImages.size());
        }

        void WindowRenderContext::CreateInstance()
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

        void WindowRenderContext::CreateSurface()
        {
            Internal::CheckResult(glfwCreateWindowSurface(m_instance, m_window.Handle(), nullptr, &m_surface), "glfwCreateWindowSurface failed");
        }

        void WindowRenderContext::SelectPhysicalDevice()
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

        void WindowRenderContext::CreateLogicalDevice()
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

        void WindowRenderContext::CreateSwapchain()
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
            createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

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

        void WindowRenderContext::CreateRenderPass()
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

            VkAttachmentReference colorAttachmentRef = {};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass = {};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;

            VkSubpassDependency dependency = {};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.srcAccessMask = 0;
            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            VkRenderPassCreateInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments = &colorAttachment;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = 1;
            renderPassInfo.pDependencies = &dependency;

            Internal::CheckResult(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass), "vkCreateRenderPass failed");
        }

        void WindowRenderContext::CreateFramebuffers()
        {
            m_framebuffers.resize(m_swapchainImageViews.size());
            for (size_t i = 0; i < m_swapchainImageViews.size(); i++)
            {
                VkFramebufferCreateInfo framebufferInfo = {};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = m_renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &m_swapchainImageViews[i];
                framebufferInfo.width = m_swapchainExtent.width;
                framebufferInfo.height = m_swapchainExtent.height;
                framebufferInfo.layers = 1;
                Internal::CheckResult(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_framebuffers[i]), "vkCreateFramebuffer failed");
            }
        }

        void WindowRenderContext::CreateCommandObjects()
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

        VkCommandBuffer WindowRenderContext::BeginSingleTimeCommands()
        {
            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = m_commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            Internal::CheckResult(vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer), "vkAllocateCommandBuffers failed");

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            return commandBuffer;
        }

        void WindowRenderContext::EndSingleTimeCommands(VkCommandBuffer commandBuffer)
        {
            Internal::CheckResult(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer failed");

            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            Internal::CheckResult(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit failed");
            Internal::CheckResult(vkQueueWaitIdle(m_graphicsQueue), "vkQueueWaitIdle failed");

            vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
        }

        void WindowRenderContext::CreateTextureResources()
        {
            VkDeviceSize textureSize = static_cast<VkDeviceSize>(m_textureWidth) * static_cast<VkDeviceSize>(m_textureHeight) * 4;

            m_stagingBuffer = Internal::CreateBuffer(
                m_physicalDevice,
                m_device,
                textureSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_stagingMemory);

            VkImageCreateInfo imageInfo = {};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent.width = static_cast<uint32_t>(m_textureWidth);
            imageInfo.extent.height = static_cast<uint32_t>(m_textureHeight);
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            Internal::CheckResult(vkCreateImage(m_device, &imageInfo, nullptr, &m_textureImage), "vkCreateImage failed");

            VkMemoryRequirements memRequirements;
            vkGetImageMemoryRequirements(m_device, m_textureImage, &memRequirements);

            VkMemoryAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = Internal::FindMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Internal::CheckResult(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_textureMemory), "vkAllocateMemory failed");
            Internal::CheckResult(vkBindImageMemory(m_device, m_textureImage, m_textureMemory, 0), "vkBindImageMemory failed");

            // Transition straight to SHADER_READ_ONLY_OPTIMAL up front, even
            // though the texture is still empty — UpdateTexture() transitions
            // back to TRANSFER_DST_OPTIMAL and back again around each upload,
            // so the pipeline can always assume SHADER_READ_ONLY_OPTIMAL
            // between frames.
            VkCommandBuffer commandBuffer = BeginSingleTimeCommands();
            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_textureImage;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            EndSingleTimeCommands(commandBuffer);

            VkImageViewCreateInfo viewInfo = {};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_textureImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            Internal::CheckResult(vkCreateImageView(m_device, &viewInfo, nullptr, &m_textureImageView), "vkCreateImageView failed");

            // Nearest filtering, not linear: this is a blocky pixel-grid
            // visualization (matching HAL's own UIGrid), not a photo — linear
            // filtering would blur cell boundaries rather than showing them
            // crisply.
            VkSamplerCreateInfo samplerInfo = {};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_NEAREST;
            samplerInfo.minFilter = VK_FILTER_NEAREST;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.maxLod = 0.0f;
            samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            Internal::CheckResult(vkCreateSampler(m_device, &samplerInfo, nullptr, &m_textureSampler), "vkCreateSampler failed");
        }

        void WindowRenderContext::CreateGraphicsPipeline()
        {
            VkDescriptorSetLayoutBinding samplerBinding = {};
            samplerBinding.binding = 0;
            samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            samplerBinding.descriptorCount = 1;
            samplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 1;
            layoutInfo.pBindings = &samplerBinding;
            Internal::CheckResult(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout), "vkCreateDescriptorSetLayout failed");

            VkDescriptorPoolSize poolSize = {};
            poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            poolSize.descriptorCount = 1;
            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            poolInfo.maxSets = 1;
            Internal::CheckResult(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool), "vkCreateDescriptorPool failed");

            VkDescriptorSetAllocateInfo setAllocInfo = {};
            setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setAllocInfo.descriptorPool = m_descriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &m_descriptorSetLayout;
            Internal::CheckResult(vkAllocateDescriptorSets(m_device, &setAllocInfo, &m_descriptorSet), "vkAllocateDescriptorSets failed");

            VkDescriptorImageInfo imageInfo = {};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = m_textureImageView;
            imageInfo.sampler = m_textureSampler;

            VkWriteDescriptorSet write = {};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = m_descriptorSet;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

            std::vector<char> vertCode = Internal::ReadFile(ShaderPath("FullscreenQuad.vert"));
            std::vector<char> fragCode = Internal::ReadFile(ShaderPath("FullscreenQuad.frag"));

            VkShaderModuleCreateInfo vertInfo = {};
            vertInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vertInfo.codeSize = vertCode.size();
            vertInfo.pCode = reinterpret_cast<const uint32_t *>(vertCode.data());
            Internal::CheckResult(vkCreateShaderModule(m_device, &vertInfo, nullptr, &m_vertShaderModule), "vkCreateShaderModule failed");

            VkShaderModuleCreateInfo fragInfo = {};
            fragInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fragInfo.codeSize = fragCode.size();
            fragInfo.pCode = reinterpret_cast<const uint32_t *>(fragCode.data());
            Internal::CheckResult(vkCreateShaderModule(m_device, &fragInfo, nullptr, &m_fragShaderModule), "vkCreateShaderModule failed");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = m_vertShaderModule;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = m_fragShaderModule;
            stages[1].pName = "main";

            VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

            VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            // Viewport and scissor are dynamic (recorded per frame from
            // the current swapchain extent) so a window resize only
            // recreates the swapchain-sized resources, never this
            // pipeline.
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

            VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo colorBlending = {};
            colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;

            VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
            Internal::CheckResult(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout), "vkCreatePipelineLayout failed");

            VkGraphicsPipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = stages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = m_pipelineLayout;
            pipelineInfo.renderPass = m_renderPass;
            pipelineInfo.subpass = 0;

            Internal::CheckResult(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline), "vkCreateGraphicsPipelines failed");
        }

        void WindowRenderContext::UpdateTexture(const std::vector<uint8_t> &pixelsRgba)
        {
            VkDeviceSize textureSize = static_cast<VkDeviceSize>(m_textureWidth) * static_cast<VkDeviceSize>(m_textureHeight) * 4;

            void *pMapped = nullptr;
            Internal::CheckResult(vkMapMemory(m_device, m_stagingMemory, 0, textureSize, 0, &pMapped), "vkMapMemory (texture staging) failed");
            std::memcpy(pMapped, pixelsRgba.data(), static_cast<size_t>(textureSize));
            vkUnmapMemory(m_device, m_stagingMemory);

            VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

            VkImageMemoryBarrier toTransferDst = {};
            toTransferDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toTransferDst.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toTransferDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toTransferDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransferDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransferDst.image = m_textureImage;
            toTransferDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toTransferDst.subresourceRange.levelCount = 1;
            toTransferDst.subresourceRange.layerCount = 1;
            toTransferDst.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            toTransferDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransferDst);

            VkBufferImageCopy region = {};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent.width = static_cast<uint32_t>(m_textureWidth);
            region.imageExtent.height = static_cast<uint32_t>(m_textureHeight);
            region.imageExtent.depth = 1;
            vkCmdCopyBufferToImage(commandBuffer, m_stagingBuffer, m_textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            VkImageMemoryBarrier toShaderRead = {};
            toShaderRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toShaderRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toShaderRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toShaderRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toShaderRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toShaderRead.image = m_textureImage;
            toShaderRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toShaderRead.subresourceRange.levelCount = 1;
            toShaderRead.subresourceRange.layerCount = 1;
            toShaderRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toShaderRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toShaderRead);

            EndSingleTimeCommands(commandBuffer);
        }

        void WindowRenderContext::CreateSyncObjects()
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

        bool WindowRenderContext::RecreateSwapchain()
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
            for (VkImageView imageView : m_swapchainImageViews)
            {
                vkDestroyImageView(m_device, imageView, nullptr);
            }
            m_swapchainImageViews.clear();
            vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
            m_swapchain = VK_NULL_HANDLE;

            CreateSwapchain();
            CreateFramebuffers();
            Logger::Instance().Log(LogSeverity::Info, "WindowRenderContext: swapchain recreated at " + std::to_string(framebufferWidth) + "x" + std::to_string(framebufferHeight));
            return true;
        }

        bool WindowRenderContext::DrawFrame()
        {
            return DrawFrame(std::function<void(VkCommandBuffer)>());
        }

        bool WindowRenderContext::DrawFrame(const std::function<void(VkCommandBuffer)> &overlay)
        {
            if (m_window.ConsumeFramebufferResized())
            {
                if (!RecreateSwapchain())
                {
                    return false;
                }
            }

            vkWaitForFences(m_device, 1, &m_inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());

            uint32_t imageIndex = 0;
            VkResult acquireResult = vkAcquireNextImageKHR(m_device, m_swapchain, std::numeric_limits<uint64_t>::max(), m_imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
            if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
            {
                // The acquire failed outright (nothing was signaled), so
                // recreate now and let the next DrawFrame draw. A
                // SUBOPTIMAL acquire, by contrast, DID succeed and signal
                // the semaphore, so that frame proceeds and present's
                // result triggers the recreation instead.
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

            // Never actually visible — the fullscreen triangle below
            // overdraws every pixel — but a render pass with a CLEAR
            // loadOp still needs a value to clear to.
            VkClearValue clearValue = {};
            clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = m_renderPass;
            renderPassInfo.framebuffer = m_framebuffers[imageIndex];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = m_swapchainExtent;
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearValue;

            vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

            VkViewport viewport = {};
            viewport.width = static_cast<float>(m_swapchainExtent.width);
            viewport.height = static_cast<float>(m_swapchainExtent.height);
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor = {};
            scissor.extent = m_swapchainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
            vkCmdDraw(commandBuffer, 3, 1, 0, 0);
            if (overlay)
            {
                overlay(commandBuffer);
            }
            vkCmdEndRenderPass(commandBuffer);

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
