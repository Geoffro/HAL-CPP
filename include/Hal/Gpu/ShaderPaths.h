#pragma once

#include <string>

namespace Hal
{
    namespace Gpu
    {
        // Returns the absolute path to a compiled shader's .spv file.
        // shaderName is the shader source's full filename, extension
        // included (e.g. "Diffusion2D.comp", "FullscreenQuad.vert") —
        // needed since two different shader stages can share a stem
        // (FullscreenQuad.vert/.frag) and would otherwise collide.
        // e.g. ShaderPath("Diffusion2D.comp") -> ".../build/shaders/Diffusion2D.comp.spv".
        // Shaders are compiled from GLSL to SPIR-V at build time (see
        // hal-cpp/CMakeLists.txt's HalShaders target) into a fixed,
        // absolute location baked in via the HAL_SHADER_DIR compile
        // definition — this never depends on the process's current
        // working directory, so it resolves correctly whether the
        // binary is run from a terminal or launched from an Xcode scheme.
        std::string ShaderPath(const std::string &shaderName);
    }
}
