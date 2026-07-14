// Ported from LEARN_HERE/Other/GuiExample: a parameter-menu demo showing
// each input widget kind (bool, double, int, string, combo-box, and a file
// path) plus a Print button that logs the current values. Java uses the
// UI*Input widgets; this uses the Dear ImGui-backed Hal::Gui::GuiContext
// (the file chooser is an editable text path — there is no native file
// dialog).
//
// Interactive only (like GuiDemo): it opens a live window and needs a
// display, so there is no headless mode. Build it and run it to try the
// controls; click Print to log the values.

#include "Hal/Color.h"
#include "Hal/Gpu/Window.h"
#include "Hal/Gpu/WindowRenderContext.h"
#include "Hal/Gui/GuiContext.h"
#include "Hal/Gui/WindowGuiRenderer.h"
#include "Hal/Log/Logger.h"

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace Hal
{
    namespace Examples
    {
        static void LogMenuValues(bool               boolean,
                                  double             dbl,
                                  int32_t            integer,
                                  const std::string &text,
                                  int32_t            optionIndex,
                                  const std::string &filePath)
        {
            std::string message = "GuiExample values:\n";
            message += "  boolean = " + std::string(boolean ? "true" : "false") + "\n";
            message += "  double  = " + std::to_string(dbl) + "\n";
            message += "  int     = " + std::to_string(integer) + "\n";
            message += "  string  = " + text + "\n";
            message += "  options = " + std::to_string(optionIndex) + "\n";
            message += "  file    = " + filePath;
            Logger::Instance().Log(LogSeverity::Info, message);
        }
    }
}

int main()
{
    try
    {
        Hal::Gpu::Window            window(500, 400, "GuiExample menu");
        Hal::Gpu::WindowRenderContext renderContext(window, 16, 16);
        Hal::Gui::GuiContext        gui;
        Hal::Gui::WindowGuiRenderer guiRenderer(window, renderContext, gui);

        bool                 boolean = false;
        double               dbl = 0.5;
        int32_t              integer = 3;
        std::string          text = "text";
        int32_t              optionIndex = 0;
        std::string          filePath = "example.txt";
        const char *const    options[] = {"option 1", "option 2", "option 3"};
        std::vector<uint8_t> background(16 * 16 * 4, 0);

        while (!window.ShouldClose())
        {
            window.PollEvents();

            guiRenderer.NewFrame();
            if (gui.BeginPanel("Example Menu"))
            {
                gui.Checkbox("boolean", boolean);
                gui.DoubleSlider("double", dbl, 0.0, 2.0);
                gui.IntSlider("int", integer, 0, 5);
                gui.StringInput("string", text);
                gui.ComboBox("options", optionIndex, options, 3);
                gui.StringInput("file chooser", filePath);
                if (gui.Button("Print"))
                {
                    Hal::Examples::LogMenuValues(boolean, dbl, integer, text, optionIndex, filePath);
                }
            }
            gui.EndPanel();

            renderContext.UpdateTexture(background);
            renderContext.DrawFrame([&guiRenderer](VkCommandBuffer commandBuffer) {
                guiRenderer.RenderOverlay(commandBuffer);
            });
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("GuiExample failed: ") + e.what());
        return 1;
    }
}
