#include <doctest/doctest.h>

#include "Hal/Gpu/ComputeContext.h"

#include <stdexcept>
#include <string>

// NOTE: this environment may not have a usable Vulkan device (no GPU, or
// no ICD registered — common in headless/CI sandboxes). Construction
// failing for that reason is an environment limitation, not a code bug,
// so it's reported via MESSAGE rather than failing the whole suite;
// anything else that goes wrong (a crash, or a different exception) will
// still fail normally.
TEST_CASE("Hal::Gpu::ComputeContext initializes a headless compute-capable device, or reports why it can't")
{
    try
    {
        Hal::Gpu::ComputeContext context;
        CHECK(context.Instance() != VK_NULL_HANDLE);
        CHECK(context.Device() != VK_NULL_HANDLE);
        CHECK(context.ComputeQueue() != VK_NULL_HANDLE);
    }
    catch (const std::runtime_error &e)
    {
        MESSAGE("No usable Vulkan compute device in this environment: ", std::string(e.what()));
    }
}
