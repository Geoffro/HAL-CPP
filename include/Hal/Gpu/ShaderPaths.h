#pragma once

#include <string>

namespace Hal
{
    namespace Gpu
    {
        // Absolute path of a compiled shader, from its full source filename:
        // ShaderPath("Diffusion2D.comp") -> "<build>/shaders/Diffusion2D.comp.spv".
        // The directory is baked in at build time (HAL_SHADER_DIR), so lookup
        // doesn't depend on the working directory.
        std::string ShaderPath(const std::string &shaderName);
    }
}
