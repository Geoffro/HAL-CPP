#include "InteractiveModel.h"

#include <GLFW/glfw3.h>

#include <cstddef>

namespace Hal
{
    namespace Examples
    {
        InteractiveModel::InteractiveModel(ITreatableTumor &model,
                                           int32_t          visX,
                                           int32_t          visY,
                                           int32_t          visScale,
                                           bool             active)
            : m_model(model),
              m_vis("Interactive Model", visX, visY, visScale, active),
              m_plot("Populations", 300, visY * visScale, 1, active),
              m_treatmentVals(model.GetTreatmentNames().size(), 0.0),
              m_plotHistory(model.GetPlotColors().size()),
              m_timeAxis(),
              m_keyWasDown(model.GetTreatmentNames().size(), false)
        {
        }

        InteractiveModel::~InteractiveModel()
        {
        }

        void InteractiveModel::PollTreatmentKeys()
        {
            void *pHandle = m_vis.NativeHandle();
            if (pHandle == nullptr)
            {
                return;
            }
            GLFWwindow *pWindow = static_cast<GLFWwindow *>(pHandle);
            for (int32_t i = 0; i < static_cast<int32_t>(m_treatmentVals.size()); i++)
            {
                bool down = glfwGetKey(pWindow, GLFW_KEY_1 + i) == GLFW_PRESS;
                if (down && !m_keyWasDown[static_cast<size_t>(i)])
                {
                    m_treatmentVals[static_cast<size_t>(i)] = (m_treatmentVals[static_cast<size_t>(i)] != 0.0) ? 0.0 : 1.0;
                }
                m_keyWasDown[static_cast<size_t>(i)] = down;
            }
        }

        void InteractiveModel::SetTreatment(int32_t treatment,
                                            double  intensity)
        {
            if ((treatment >= 0) && (treatment < static_cast<int32_t>(m_treatmentVals.size())))
            {
                m_treatmentVals[static_cast<size_t>(treatment)] = intensity;
            }
        }

        void InteractiveModel::Step(int32_t step)
        {
            m_model.InteractiveStep(m_treatmentVals, step);
            m_model.Draw(m_vis);

            std::vector<double> vals = m_model.GetPlotVals();
            m_timeAxis.push_back(static_cast<double>(step));
            for (size_t i = 0; (i < m_plotHistory.size()) && (i < vals.size()); i++)
            {
                m_plotHistory[i].push_back(vals[i]);
            }
        }

        void InteractiveModel::RefreshPlot()
        {
            std::vector<int32_t> colors = m_model.GetPlotColors();
            for (size_t i = 0; i < m_plotHistory.size(); i++)
            {
                int32_t color = (i < colors.size()) ? colors[i] : 0;
                m_plot.AddLine(color, m_timeAxis, m_plotHistory[i]);
            }
            m_plot.Render(0);
        }

        void InteractiveModel::Present(int32_t pauseMillis)
        {
            RefreshPlot();
            m_vis.TickPause(pauseMillis);
        }

        bool InteractiveModel::IsClosed() const
        {
            return m_vis.IsClosed();
        }

        GridWindow &InteractiveModel::Vis()
        {
            return m_vis;
        }
    }
}
