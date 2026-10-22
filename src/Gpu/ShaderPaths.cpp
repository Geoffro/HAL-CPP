#include "Hal/Gpu/ShaderPaths.h"

#ifndef HAL_SHADER_DIR
#error "HAL_SHADER_DIR must be defined by the build system (see CMakeLists.txt)"
#endif

namespace Hal
{
    namespace Gpu
    {
        std::string ShaderPath(const std::string &shaderName)
        {
            return std::string(HAL_SHADER_DIR) + "/" + shaderName + ".spv";
        }
    }
}
