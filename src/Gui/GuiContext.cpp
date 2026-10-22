#include "Hal/Gui/GuiContext.h"

#include "imgui.h"
#include "implot.h"

#include <cstdint>
#include <cstring>

namespace Hal
{
    namespace Gui
    {
        GuiContext::GuiContext()
            : m_pImGuiContext(nullptr),
              m_pImPlotContext(nullptr)
        {
            IMGUI_CHECKVERSION();
            m_pImGuiContext = ImGui::CreateContext();
            m_pImPlotContext = ImPlot::CreateContext();
            MakeCurrent();

            ImGuiIO &io = ImGui::GetIO();
            // No on-disk imgui.ini / log files — this is a library context,
            // not an app owning the user's home directory.
            io.IniFilename = nullptr;
            io.LogFilename = nullptr;

            // Build the font atlas up front (with a placeholder texture id)
            // so NewFrame works even with no rendering backend attached —
            // the headless test path relies on this. A real renderer
            // uploads the atlas and replaces the id.
            unsigned char *pPixels = nullptr;
            int width = 0;
            int height = 0;
            io.Fonts->GetTexDataAsRGBA32(&pPixels, &width, &height);
            io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(1)));
        }

        GuiContext::~GuiContext()
        {
            ImPlot::DestroyContext(m_pImPlotContext);
            ImGui::DestroyContext(m_pImGuiContext);
        }

        void GuiContext::MakeCurrent()
        {
            ImGui::SetCurrentContext(m_pImGuiContext);
            ImPlot::SetCurrentContext(m_pImPlotContext);
        }

        void GuiContext::BeginFrame(int32_t displayWidth,
                                    int32_t displayHeight,
                                    double  deltaTime)
        {
            MakeCurrent();
            ImGuiIO &io = ImGui::GetIO();
            io.DisplaySize = ImVec2(static_cast<float>(displayWidth), static_cast<float>(displayHeight));
            io.DeltaTime = (deltaTime > 0.0) ? static_cast<float>(deltaTime) : (1.0f / 60.0f);
            ImGui::NewFrame();
        }

        void GuiContext::NewFrameRaw()
        {
            MakeCurrent();
            ImGui::NewFrame();
        }

        void GuiContext::EndFrame()
        {
            ImGui::Render();
        }

        ImDrawData *GuiContext::DrawData() const
        {
            return ImGui::GetDrawData();
        }

        bool GuiContext::BeginPanel(const char *pTitle)
        {
            return ImGui::Begin(pTitle);
        }

        void GuiContext::EndPanel()
        {
            ImGui::End();
        }

        void GuiContext::Text(const char *pText)
        {
            ImGui::TextUnformatted(pText);
        }

        bool GuiContext::Checkbox(const char *pLabel,
                                  bool       &value)
        {
            return ImGui::Checkbox(pLabel, &value);
        }

        bool GuiContext::IntSlider(const char *pLabel,
                                   int32_t    &value,
                                   int32_t     minValue,
                                   int32_t     maxValue)
        {
            return ImGui::SliderInt(pLabel, &value, minValue, maxValue);
        }

        bool GuiContext::DoubleSlider(const char *pLabel,
                                      double     &value,
                                      double      minValue,
                                      double      maxValue)
        {
            return ImGui::SliderScalar(pLabel, ImGuiDataType_Double, &value, &minValue, &maxValue);
        }

        bool GuiContext::Button(const char *pLabel)
        {
            return ImGui::Button(pLabel);
        }

        bool GuiContext::StringInput(const char  *pLabel,
                                     std::string &value)
        {
            char buffer[256];
            std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
            buffer[sizeof(buffer) - 1] = '\0';
            if (ImGui::InputText(pLabel, buffer, sizeof(buffer)))
            {
                value = buffer;
                return true;
            }
            return false;
        }

        bool GuiContext::ComboBox(const char        *pLabel,
                                  int32_t           &current,
                                  const char *const *pItems,
                                  int32_t            count)
        {
            return ImGui::Combo(pLabel, &current, pItems, count);
        }

        bool GuiContext::BeginPlot(const char *pTitle,
                                   const char *pXLabel,
                                   const char *pYLabel)
        {
            if (!ImPlot::BeginPlot(pTitle))
            {
                return false;
            }
            ImPlot::SetupAxes(pXLabel, pYLabel);
            return true;
        }

        void GuiContext::PlotLine(const char   *pLabel,
                                  const double *pXValues,
                                  const double *pYValues,
                                  int32_t       count)
        {
            ImPlot::PlotLine(pLabel, pXValues, pYValues, count);
        }

        void GuiContext::PlotScatter(const char   *pLabel,
                                     const double *pXValues,
                                     const double *pYValues,
                                     int32_t       count)
        {
            ImPlot::PlotScatter(pLabel, pXValues, pYValues, count);
        }

        void GuiContext::EndPlot()
        {
            ImPlot::EndPlot();
        }
    }
}
