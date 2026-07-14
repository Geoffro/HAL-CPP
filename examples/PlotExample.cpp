// Ported from LEARN_HERE/Other/PlotExample: the minimal plot demo — a
// single red polyline through a handful of hand-picked points.
//
// Always writes plotexample.png; also shows a window unless run with
// "headless".

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"

#include "common/PlotWindow.h"

#include <cstring>
#include <exception>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        std::vector<double> xData = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> yData = {2.0, 1.0, 2.0, 2.0, 3.0, 6.0};

        Hal::Examples::PlotWindow plot("Plot Example", 250, 250, 2, !headless);
        plot.AddLine(Hal::RED, xData, yData);
        plot.Render(Hal::BLACK);
        plot.ToPng("plotexample.png");

        if (!headless)
        {
            plot.Show();
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("PlotExample failed: ") + e.what());
        return 1;
    }
}
