#include <doctest/doctest.h>

#include "GpuTestSupport.h"

#include "Hal/Gpu/ComputeContext.h"

#include <memory>

TEST_CASE("Hal::Gpu::ComputeContext initializes a headless compute-capable device")
{
    std::unique_ptr<Hal::Gpu::ComputeContext> pContext = TestHelpers::TryCreateComputeContext();
    if (pContext == nullptr)
    {
        return;
    }
    Hal::Gpu::ComputeContext &context = *pContext;
    CHECK(context.Instance() != VK_NULL_HANDLE);
    CHECK(context.Device() != VK_NULL_HANDLE);
    CHECK(context.ComputeQueue() != VK_NULL_HANDLE);
}
