#include "Hal/Gpu/GpuDiffusionBackend2D.h"

#include <stdexcept>

namespace Hal
{
    namespace Gpu
    {
        GpuDiffusionBackend2D::GpuDiffusionBackend2D(ComputeContext &context,
                                                     int32_t         xDim,
                                                     int32_t         yDim)
            : m_xDim(xDim),
              m_yDim(yDim),
              m_kernel(context, xDim, yDim)
        {
        }

        GpuDiffusionBackend2D::~GpuDiffusionBackend2D()
        {
        }

        void GpuDiffusionBackend2D::Diffuse(const std::vector<double> &field,
                                            std::vector<double>       &deltas,
                                            double                     diffRate,
                                            int32_t                    xDim,
                                            int32_t                    yDim,
                                            bool                       wrapX,
                                            bool                       wrapY)
        {
            if ((xDim != m_xDim) || (yDim != m_yDim))
            {
                throw std::invalid_argument("GpuDiffusionBackend2D::Diffuse called with dimensions that don't match its fixed-size kernel");
            }
            Hal::Internal::CheckFieldSizes(field, deltas, xDim, yDim, "GpuDiffusionBackend2D::Diffuse");

            size_t length = static_cast<size_t>(xDim) * static_cast<size_t>(yDim);
            std::vector<float> floatField(length);
            for (size_t i = 0; i < length; i++)
            {
                floatField[i] = static_cast<float>(field[i]);
            }

            m_kernel.UploadField(floatField);
            m_kernel.ClearDeltas();
            m_kernel.Dispatch(static_cast<float>(diffRate), wrapX, wrapY);
            std::vector<float> floatDeltas = m_kernel.DownloadDeltas();

            for (size_t i = 0; i < length; i++)
            {
                deltas[i] += static_cast<double>(floatDeltas[i]);
            }
        }
    }
}
