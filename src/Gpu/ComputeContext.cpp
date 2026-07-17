#include "Hal/Gpu/ComputeContext.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Log/Logger.h"

#include "Hal/Gpu/Internal/VmaUsage.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace Hal
{
    namespace Gpu
    {
        ComputeContext::ComputeContext()
            : m_instance(VK_NULL_HANDLE),
              m_physicalDevice(VK_NULL_HANDLE),
              m_device(VK_NULL_HANDLE),
              m_computeQueue(VK_NULL_HANDLE),
              m_computeQueueFamilyIndex(0),
              m_allocator(VK_NULL_HANDLE)
        {
            Internal::CheckResult(volkInitialize(), "volkInitialize failed — is the Vulkan loader installed?");
            CreateInstance();
            volkLoadInstance(m_instance);
            SelectPhysicalDevice();
            CreateLogicalDevice();
            volkLoadDevice(m_device);
            vkGetDeviceQueue(m_device, m_computeQueueFamilyIndex, 0, &m_computeQueue);
            CreateAllocator();
            Logger::Instance().Log(LogSeverity::Info, "Hal::Gpu::ComputeContext initialized");
        }

        ComputeContext::~ComputeContext()
        {
            if (m_allocator != VK_NULL_HANDLE)
            {
                vmaDestroyAllocator(m_allocator);
            }
            if (m_device != VK_NULL_HANDLE)
            {
                vkDestroyDevice(m_device, nullptr);
            }
            if (m_instance != VK_NULL_HANDLE)
            {
                vkDestroyInstance(m_instance, nullptr);
            }
        }

        void ComputeContext::CreateAllocator()
        {
            // volk loads Vulkan dynamically, so VMA is built with
            // VMA_DYNAMIC_VULKAN_FUNCTIONS and handed volk's two loader
            // entry points to fetch everything else itself.
            VmaVulkanFunctions vulkanFunctions = {};
            vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
            vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

            VmaAllocatorCreateInfo allocatorInfo = {};
            allocatorInfo.instance = m_instance;
            allocatorInfo.physicalDevice = m_physicalDevice;
            allocatorInfo.device = m_device;
            allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_0;
            allocatorInfo.pVulkanFunctions = &vulkanFunctions;
            Internal::CheckResult(vmaCreateAllocator(&allocatorInfo, &m_allocator), "vmaCreateAllocator failed");
        }

        VmaAllocator ComputeContext::Allocator() const
        {
            return m_allocator;
        }

        VkInstance ComputeContext::Instance() const
        {
            return m_instance;
        }

        VkPhysicalDevice ComputeContext::PhysicalDevice() const
        {
            return m_physicalDevice;
        }

        VkDevice ComputeContext::Device() const
        {
            return m_device;
        }

        VkQueue ComputeContext::ComputeQueue() const
        {
            return m_computeQueue;
        }

        uint32_t ComputeContext::ComputeQueueFamilyIndex() const
        {
            return m_computeQueueFamilyIndex;
        }

        void ComputeContext::CreateInstance()
        {
            uint32_t availableCount = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &availableCount, nullptr);
            std::vector<VkExtensionProperties> available(availableCount);
            vkEnumerateInstanceExtensionProperties(nullptr, &availableCount, available.data());

            std::vector<const char *> enabledExtensions;
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

        void ComputeContext::SelectPhysicalDevice()
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

                for (uint32_t i = 0; i < queueFamilyCount; i++)
                {
                    if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0)
                    {
                        m_physicalDevice = candidate;
                        m_computeQueueFamilyIndex = i;
                        return;
                    }
                }
            }

            Logger::Instance().Log(LogSeverity::Error, "No physical device with a compute-capable queue family found");
            throw std::runtime_error("no physical device with a compute-capable queue family found");
        }

        void ComputeContext::CreateLogicalDevice()
        {
            const float queuePriority = 1.0f;
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = m_computeQueueFamilyIndex;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;

            uint32_t availableCount = 0;
            vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &availableCount, nullptr);
            std::vector<VkExtensionProperties> available(availableCount);
            vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &availableCount, available.data());

            std::vector<const char *> enabledExtensions;
            if (Internal::HasDeviceExtension(available, "VK_KHR_portability_subset"))
            {
                enabledExtensions.push_back("VK_KHR_portability_subset");
            }

            VkDeviceCreateInfo deviceCreateInfo = {};
            deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            deviceCreateInfo.queueCreateInfoCount = 1;
            deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
            deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
            deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.empty() ? nullptr : enabledExtensions.data();

            Internal::CheckResult(vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device), "vkCreateDevice failed");
        }
    }
}
