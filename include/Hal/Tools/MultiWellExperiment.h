#pragma once

#include "Hal/Color.h"
#include "Hal/Concurrency/MultiThread.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Hal
{
    // Runs many independent model instances ("wells") and composites
    // their rendered pixels into one RGBA8 image with 1px borders between
    // wells, for ExportPng, VideoWriter, or a WindowRenderContext texture.
    // Wells fill each column bottom to top, then move right; y is up, as in
    // RenderGrid2DToRgba. TModel is the per-well model type; the caller
    // owns the models.
    //   stepper: void(TModel &model, int32_t wellIndex)
    //   drawer:  int32_t(const TModel &model, int32_t x, int32_t y)
    //            returning a packed ARGB color
    template <typename TModel>
    class MultiWellExperiment
    {
    public:
        MultiWellExperiment(int32_t                       wellsX,
                            int32_t                       wellsY,
                            int32_t                       visXdim,
                            int32_t                       visYdim,
                            int32_t                       borderColor,
                            const std::vector<TModel *>  &wells)
            : m_wellsX(wellsX),
              m_wellsY(wellsY),
              m_visXdim(visXdim),
              m_visYdim(visYdim),
              m_borderColor(borderColor),
              m_wells(wells),
              m_compositeWidth(visXdim * wellsX + (wellsX - 1)),
              m_compositeHeight(visYdim * wellsY + (wellsY - 1)),
              m_pixels(static_cast<size_t>(m_compositeWidth) * static_cast<size_t>(m_compositeHeight) * 4, 0)
        {
            if (static_cast<int32_t>(wells.size()) > (wellsX * wellsY))
            {
                throw std::invalid_argument("MultiWellExperiment given more wells than the wellsX*wellsY layout can hold");
            }
        }

        ~MultiWellExperiment()
        {
        }

        MultiWellExperiment(const MultiWellExperiment &other) = delete;
        MultiWellExperiment &operator=(const MultiWellExperiment &other) = delete;

        int32_t CompositeWidth() const
        {
            return m_compositeWidth;
        }

        int32_t CompositeHeight() const
        {
            return m_compositeHeight;
        }

        // The composited RGBA8 buffer (row-major, image row 0 at the top),
        // valid after Render().
        const std::vector<uint8_t> &Pixels() const
        {
            return m_pixels;
        }

        // Replaces the wells (e.g. to reset an experiment), keeping the same
        // layout.
        void LoadWells(const std::vector<TModel *> &wells)
        {
            if (static_cast<int32_t>(wells.size()) > (m_wellsX * m_wellsY))
            {
                throw std::invalid_argument("MultiWellExperiment given more wells than the wellsX*wellsY layout can hold");
            }
            m_wells = wells;
        }

        // Advances every well one step, serially.
        template <typename StepperFn>
        void Step(StepperFn stepper)
        {
            for (int32_t i = 0; i < static_cast<int32_t>(m_wells.size()); i++)
            {
                stepper(*m_wells[static_cast<size_t>(i)], i);
            }
        }

        // Advances every well one step, distributing wells across threads
        // (via Hal::MultiThread). The stepper must be safe to call
        // concurrently across distinct wells (independent models are).
        template <typename StepperFn>
        void StepMultiThread(StepperFn stepper)
        {
            int32_t wellCount = static_cast<int32_t>(m_wells.size());
            MultiThread(wellCount, [&](int32_t i)
            {
                stepper(*m_wells[static_cast<size_t>(i)], i);
            });
        }

        // Composites every well's pixels into the buffer, clearing to the
        // border color first (so gaps and unused layout cells show the
        // border).
        template <typename DrawerFn>
        void Render(DrawerFn drawer)
        {
            uint8_t borderR = static_cast<uint8_t>(GetRed256(m_borderColor));
            uint8_t borderG = static_cast<uint8_t>(GetGreen256(m_borderColor));
            uint8_t borderB = static_cast<uint8_t>(GetBlue256(m_borderColor));
            uint8_t borderA = static_cast<uint8_t>(GetAlpha256(m_borderColor));
            for (size_t i = 0; i < m_pixels.size(); i += 4)
            {
                m_pixels[i] = borderR;
                m_pixels[i + 1] = borderG;
                m_pixels[i + 2] = borderB;
                m_pixels[i + 3] = borderA;
            }

            for (int32_t well = 0; well < static_cast<int32_t>(m_wells.size()); well++)
            {
                // The +1 leaves a 1px border between wells.
                int32_t xWell = well / m_wellsY;
                int32_t yWell = well % m_wellsY;
                int32_t xStart = xWell * (m_visXdim + 1);
                int32_t yStart = yWell * (m_visYdim + 1);
                for (int32_t x = 0; x < m_visXdim; x++)
                {
                    for (int32_t y = 0; y < m_visYdim; y++)
                    {
                        int32_t color = drawer(*m_wells[static_cast<size_t>(well)], x, y);
                        SetCompositePixel(xStart + x, yStart + y, color);
                    }
                }
            }
        }

    private:
        void SetCompositePixel(int32_t compositeX,
                               int32_t compositeY,
                               int32_t color)
        {
            int32_t imageRow = m_compositeHeight - 1 - compositeY;
            size_t  index = (static_cast<size_t>(imageRow) * static_cast<size_t>(m_compositeWidth) + static_cast<size_t>(compositeX)) * 4;
            m_pixels[index] = static_cast<uint8_t>(GetRed256(color));
            m_pixels[index + 1] = static_cast<uint8_t>(GetGreen256(color));
            m_pixels[index + 2] = static_cast<uint8_t>(GetBlue256(color));
            m_pixels[index + 3] = static_cast<uint8_t>(GetAlpha256(color));
        }

        int32_t m_wellsX;
        int32_t m_wellsY;
        int32_t m_visXdim;
        int32_t m_visYdim;
        int32_t m_borderColor;
        std::vector<TModel *> m_wells;
        int32_t m_compositeWidth;
        int32_t m_compositeHeight;
        std::vector<uint8_t> m_pixels;
    };
}
