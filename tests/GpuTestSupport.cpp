#include "GpuTestSupport.h"

#include <doctest/doctest.h>

#include <stdexcept>

namespace TestHelpers
{
    std::unique_ptr<Hal::Gpu::ComputeContext> TryCreateComputeContext()
    {
        try
        {
            return std::make_unique<Hal::Gpu::ComputeContext>();
        }
        catch (const std::runtime_error &e)
        {
            MESSAGE("Skipped: no usable Vulkan compute device: ", std::string(e.what()));
            return nullptr;
        }
    }

#ifdef HAL_BUILD_WINDOW
    std::unique_ptr<Hal::Gpu::Window> TryCreateWindow(int32_t            width,
                                                      int32_t            height,
                                                      const std::string &title)
    {
        if (TryCreateComputeContext() == nullptr)
        {
            return nullptr;
        }
        try
        {
            return std::make_unique<Hal::Gpu::Window>(width, height, title);
        }
        catch (const std::runtime_error &e)
        {
            MESSAGE("Skipped: no usable windowing system: ", std::string(e.what()));
            return nullptr;
        }
    }
#endif
}
