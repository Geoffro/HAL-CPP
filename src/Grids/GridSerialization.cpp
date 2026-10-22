#include "Hal/Grids/GridSerialization.h"

#include "Hal/Log/Logger.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace Hal
{
    namespace Internal
    {
        void CheckLoadedDims(const BinaryReader &reader,
                             int32_t             xDim,
                             int32_t             yDim,
                             int32_t             zDim,
                             const char         *pWhat)
        {
            // The second product only runs once xDim * yDim fits in int32_t,
            // so it can't overflow int64_t.
            int64_t maxLength = std::numeric_limits<int32_t>::max();
            bool    valid = reader.IsGood() && (xDim > 0) && (yDim > 0) && (zDim > 0) &&
                            ((static_cast<int64_t>(xDim) * yDim) <= maxLength) &&
                            ((static_cast<int64_t>(xDim) * yDim * zDim) <= maxLength);
            if (!valid)
            {
                std::string message = std::string(pWhat) + ": failed read or invalid dimensions in saved data";
                Logger::Instance().Log(LogSeverity::Error, message);
                throw std::runtime_error(message);
            }
        }
    }
}
