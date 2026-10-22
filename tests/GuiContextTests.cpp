#include <doctest/doctest.h>

#include "Hal/Gui/GuiContext.h"

#include "imgui.h"

#include <cstdint>
#include <string>
#include <vector>

// Headless: frames are built with no window or GPU backend, and the tests
// check that geometry is produced and controls pass values through. ImGui
// hides an auto-sized window on its first frame, so each test builds a few
// frames before asserting.

TEST_CASE("Hal::Gui::GuiContext builds a frame of controls and produces draw data")
{
    Hal::Gui::GuiContext gui;

    bool boolValue = true;
    int32_t intValue = 5;
    double doubleValue = 0.5;

    ImDrawData *pDrawData = nullptr;
    for (int32_t frame = 0; frame < 3; frame++)
    {
        gui.BeginFrame(800, 600, 1.0 / 60.0);
        if (gui.BeginPanel("Parameters"))
        {
            gui.Text("controls");
            gui.Checkbox("enabled", boolValue);
            gui.IntSlider("count", intValue, 0, 10);
            gui.DoubleSlider("rate", doubleValue, 0.0, 1.0);
            gui.Button("run");
        }
        gui.EndPanel();
        gui.EndFrame();
        pDrawData = gui.DrawData();
    }

    REQUIRE(pDrawData != nullptr);
    CHECK(pDrawData->Valid);
    CHECK(pDrawData->CmdListsCount > 0);
    CHECK(pDrawData->TotalVtxCount > 0);
    CHECK(pDrawData->TotalIdxCount > 0);

    // With no simulated input the bound values are unchanged.
    CHECK(boolValue == true);
    CHECK(intValue == 5);
    CHECK(doubleValue == doctest::Approx(0.5));
}

TEST_CASE("Hal::Gui::GuiContext ImPlot line and scatter plots produce geometry")
{
    Hal::Gui::GuiContext gui;

    std::vector<double> xs = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> ysLine = {0.0, 1.0, 4.0, 9.0, 16.0};
    std::vector<double> ysScatter = {1.0, 0.5, 0.25, 0.125, 0.0625};
    int32_t count = static_cast<int32_t>(xs.size());

    ImDrawData *pDrawData = nullptr;
    for (int32_t frame = 0; frame < 3; frame++)
    {
        gui.BeginFrame(1024, 768, 1.0 / 60.0);
        if (gui.BeginPanel("Plots"))
        {
            if (gui.BeginPlot("metrics", "time", "value"))
            {
                gui.PlotLine("squared", xs.data(), ysLine.data(), count);
                gui.PlotScatter("decay", xs.data(), ysScatter.data(), count);
                gui.EndPlot();
            }
        }
        gui.EndPanel();
        gui.EndFrame();
        pDrawData = gui.DrawData();
    }

    REQUIRE(pDrawData != nullptr);
    CHECK(pDrawData->Valid);
    CHECK(pDrawData->TotalVtxCount > 0);
}

TEST_CASE("Hal::Gui::GuiContext supports repeated frames and a fresh context each time")
{
    for (int32_t iteration = 0; iteration < 3; iteration++)
    {
        Hal::Gui::GuiContext gui;
        ImDrawData *pDrawData = nullptr;
        for (int32_t frame = 0; frame < 5; frame++)
        {
            gui.BeginFrame(640, 480, 1.0 / 60.0);
            bool flag = (frame % 2) == 0;
            if (gui.BeginPanel("Repeated"))
            {
                gui.Checkbox("flag", flag);
            }
            gui.EndPanel();
            gui.EndFrame();
            pDrawData = gui.DrawData();
            REQUIRE(pDrawData != nullptr);
            CHECK(pDrawData->Valid);
        }
        // By the last frame the window is measured and visible.
        CHECK(pDrawData->TotalVtxCount > 0);
    }
}

TEST_CASE("Hal::Gui::GuiContext configures its own ImGui context even when another is current")
{
    Hal::Gui::GuiContext first;
    Hal::Gui::GuiContext second;
    second.MakeCurrent();
    CHECK(ImGui::GetIO().IniFilename == nullptr);
    CHECK(ImGui::GetIO().LogFilename == nullptr);
}

TEST_CASE("Hal::Gui::GuiContext text and combo controls build without input")
{
    Hal::Gui::GuiContext gui;
    std::string text = "hello";
    int32_t choice = 1;
    const char *items[3] = {"a", "b", "c"};
    for (int32_t frame = 0; frame < 3; frame++)
    {
        gui.BeginFrame(640, 480, 1.0 / 60.0);
        if (gui.BeginPanel("Inputs"))
        {
            CHECK_FALSE(gui.StringInput("Name", text));
            CHECK_FALSE(gui.ComboBox("Pick", choice, items, 3));
        }
        gui.EndPanel();
        gui.EndFrame();
    }
    CHECK(text == "hello");
    CHECK(choice == 1);
    CHECK(gui.DrawData()->TotalVtxCount > 0);
}
