#pragma once

#include <cstdint>
#include <string>

struct ImDrawData;
struct ImGuiContext;
struct ImPlotContext;

namespace Hal
{
    namespace Gui
    {
        // Owns a Dear ImGui + ImPlot context and provides the HAL GUI API
        // shape — parameter controls (replacing Java's UIBoolInput /
        // UIIntInput / UIDoubleInput) and line/scatter plots (the UIPlot
        // equivalent). This is the platform-agnostic core: it uses only
        // ImGui/ImPlot core, not their GLFW/Vulkan backends, so a frame can
        // be built and its draw data inspected with no window (that path is
        // unit-tested headlessly). The GLFW input + Vulkan rendering
        // integration that turns this into an on-screen UI lives in
        // HalWindow.
        //
        // Usage per frame: BeginFrame -> build panels/controls/plots ->
        // EndFrame, then a renderer consumes DrawData(). The control helpers
        // are immediate-mode: they bind directly to a caller-owned value and
        // return whether it changed this frame.
        //
        // Not reentrant / single-instance: ImGui keeps a global "current
        // context", which BeginFrame sets — use one GuiContext at a time.
        class GuiContext
        {
        public:
            GuiContext();
            ~GuiContext();

            GuiContext(const GuiContext &other) = delete;
            GuiContext &operator=(const GuiContext &other) = delete;

            // Makes this context's ImGui + ImPlot the current context.
            // Needed before calling ImGui platform/renderer backend
            // functions that operate on the global current context (see
            // Hal::Gui::WindowGuiRenderer).
            void MakeCurrent();

            // Starts a new UI frame for a display of the given pixel size.
            // deltaTime is the seconds since the last frame (used by ImGui
            // for input timing/animation; a sensible default is substituted
            // if <= 0). This is the headless entry point — it sets the
            // display size / delta time itself.
            void BeginFrame(int32_t displayWidth,
                            int32_t displayHeight,
                            double  deltaTime);

            // Starts a new UI frame assuming a platform backend (e.g.
            // imgui_impl_glfw) has already set the display size / delta time
            // / input on the IO. Used by the windowed renderer.
            void NewFrameRaw();

            // Finalizes the frame's geometry (call after building the UI).
            void EndFrame();

            // The finalized draw data from the most recent EndFrame, for a
            // renderer to submit. Valid until the next BeginFrame.
            ImDrawData *DrawData() const;

            // Begins a titled panel/window; returns false if it is collapsed
            // (skip building its contents then). Always pair with EndPanel.
            bool BeginPanel(const char *pTitle);
            void EndPanel();

            void Text(const char *pText);

            // Parameter controls. Each binds to a caller-owned value and
            // returns whether the user changed it this frame.
            bool Checkbox(const char *pLabel,
                          bool       &value);
            bool IntSlider(const char *pLabel,
                           int32_t    &value,
                           int32_t     minValue,
                           int32_t     maxValue);
            bool DoubleSlider(const char *pLabel,
                              double     &value,
                              double      minValue,
                              double      maxValue);
            bool Button(const char *pLabel);

            // Single-line text field (replacing Java's UIStringInput /
            // UIFileChooserInput — the latter without a native file dialog,
            // just an editable path). Returns true when edited. The value is
            // capped at 255 characters.
            bool StringInput(const char  *pLabel,
                             std::string &value);

            // Dropdown selecting one of `count` items (replacing Java's
            // UIComboBoxInput). `current` is the selected index; returns true
            // when it changes.
            bool ComboBox(const char        *pLabel,
                          int32_t           &current,
                          const char *const *pItems,
                          int32_t            count);

            // Plotting (ImPlot). BeginPlot returns false if the plot is not
            // visible (skip the Plot* calls then); pair with EndPlot.
            bool BeginPlot(const char *pTitle,
                           const char *pXLabel,
                           const char *pYLabel);
            void PlotLine(const char   *pLabel,
                          const double *pXValues,
                          const double *pYValues,
                          int32_t       count);
            void PlotScatter(const char   *pLabel,
                             const double *pXValues,
                             const double *pYValues,
                             int32_t       count);
            void EndPlot();

        private:
            ImGuiContext *m_pImGuiContext;
            ImPlotContext *m_pImPlotContext;
        };
    }
}
