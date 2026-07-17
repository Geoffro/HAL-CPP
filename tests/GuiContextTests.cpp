#include <doctest/doctest.h>

#include "Hal/Gui/GuiContext.h"

#include "imgui.h"

#include <cstdint>
#include <vector>

// These run headlessly: the GUI core builds ImGui/ImPlot frames and
// produces draw data with no window or GPU backend attached (the context
// builds the font atlas up front so NewFrame works). The tests verify the
// integration is wired correctly by checking that real geometry is
// produced and that the controls pass values through, rather than
// simulating pixel-level clicks (which ImGui's multi-frame input state
// machine makes brittle to drive from a test).
//
// NOTE: ImGui hides an auto-sized window on its first frame while it
// measures the content, so geometry only appears from the second frame on
// — the tests build a few frames before asserting, as a real render loop
// would.

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
