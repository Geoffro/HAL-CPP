#include "Hal/Gpu/GpuAdiDiffusionBackend2D.h"

#include "Hal/Pde/IDiffusionBackend2D.h"

#include <stdexcept>

namespace Hal
{
    namespace Gpu
    {
        GpuAdiDiffusionBackend2D::GpuAdiDiffusionBackend2D(ComputeContext &context,
                                                           int32_t         xDim,
                                                           int32_t         yDim)
            : m_xDim(xDim),
              m_yDim(yDim),
              m_kernel(context, xDim, yDim)
        {
        }

        GpuAdiDiffusionBackend2D::~GpuAdiDiffusionBackend2D()
        {
        }

        void GpuAdiDiffusionBackend2D::DiffuseADI(const std::vector<double> &field,
                                                  std::vector<double>       &deltas,
                                                  double                     diffRate,
                                                  int32_t                    xDim,
                                                  int32_t                    yDim,
                                                  bool                       wrapX,
                                                  bool                       wrapY)
        {
            if ((xDim != m_xDim) || (yDim != m_yDim))
            {
                throw std::invalid_argument("GpuAdiDiffusionBackend2D::DiffuseADI called with dimensions differing from construction");
            }
            Hal::Internal::CheckFieldSizes(field, deltas, xDim, yDim, "GpuAdiDiffusionBackend2D::DiffuseADI");

            std::vector<float> floatField(field.size());
            for (size_t k = 0; k < field.size(); k++)
            {
                floatField[k] = static_cast<float>(field[k]);
            }

            m_kernel.RunSteps(floatField, 1, static_cast<float>(diffRate), wrapX, wrapY);

            for (size_t k = 0; k < field.size(); k++)
            {
                deltas[k] += static_cast<double>(floatField[k]) - field[k];
            }
        }
    }
}
