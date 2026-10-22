#include "Hal/Pde/IDiffusionBackend2D.h"

#include "Hal/Log/Logger.h"

#include <stdexcept>
#include <string>

namespace Hal
{
    IDiffusionBackend2D::IDiffusionBackend2D()
    {
    }

    IDiffusionBackend2D::~IDiffusionBackend2D()
    {
    }

    namespace Internal
    {
        void CheckFieldSizes(const std::vector<double> &field,
                             const std::vector<double> &deltas,
                             int32_t                    xDim,
                             int32_t                    yDim,
                             const char                *pCaller)
        {
            size_t length = static_cast<size_t>(xDim) * static_cast<size_t>(yDim);
            if ((xDim <= 0) || (yDim <= 0) || (field.size() != length) || (deltas.size() != length))
            {
                std::string message = std::string(pCaller) + ": field and deltas must each hold xDim*yDim values";
                Logger::Instance().Log(LogSeverity::Error, message);
                throw std::invalid_argument(message);
            }
        }
    }
}
