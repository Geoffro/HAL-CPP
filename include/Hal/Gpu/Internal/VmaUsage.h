#pragma once

// Single include point for VMA's header: suppresses the third-party
// nullability warnings clang otherwise emits for every translation unit
// that touches vk_mem_alloc.h. Hal::Gpu .cpp files include this instead
// of <vk_mem_alloc.h> directly (public headers only forward-declare the
// VMA handles — see ComputeContext.h).

#include <volk.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#endif

#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
