// The single translation unit compiling VMA's implementation (the same
// one-definition convention as the vendored stb/jo_mpeg headers — see
// src/Video/StbImageWriteImpl.cpp). Every other Hal::Gpu .cpp just
// includes <vk_mem_alloc.h> for the declarations.

#define VMA_IMPLEMENTATION
#include "Hal/Gpu/Internal/VmaUsage.h"
