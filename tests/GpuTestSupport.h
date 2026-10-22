#pragma once

#include "Hal/Gpu/ComputeContext.h"

#include <cstdint>
#include <memory>
#include <string>

#ifdef HAL_BUILD_WINDOW
#include "Hal/Gpu/Window.h"
#endif

namespace TestHelpers
{
    // Creates a ComputeContext, or returns null after a doctest MESSAGE if
    // this environment has no usable Vulkan device. Environment setup is the
    // only skip point for GPU tests; any failure after it fails the test.
    std::unique_ptr<Hal::Gpu::ComputeContext> TryCreateComputeContext();

#ifdef HAL_BUILD_WINDOW
    // Creates a Window, or returns null after a doctest MESSAGE if this
    // environment has no Vulkan device or no display.
    std::unique_ptr<Hal::Gpu::Window> TryCreateWindow(int32_t            width,
                                                      int32_t            height,
                                                      const std::string &title);
#endif
}
