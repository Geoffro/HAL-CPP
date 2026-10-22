// A two-variable predator-prey style tumor/immune ODE integrated with the
// adaptive Runge-Kutta-Fehlberg solver (Runge45Record keeps the whole
// trajectory), plotted as tumor (green) and immune (blue) curves over
// time.
//
// Always writes tumorimmunemodel.png; also shows a window unless run with
// "headless".

#include "Hal/Color.h"
#include "Hal/Log/Logger.h"
#include "Hal/Tools/OdeSolver.h"

#include "common/PlotWindow.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static const int32_t TUMOR = 0;
static const int32_t IMMUNE = 1;

struct TumorGrowth
{
    double tumorGrowthRate;
    double immuneDeathRate;
    double immuneKillTumorRate;
    double immuneRecruitmentRate;

    void Derivatives(double /*t*/,
                     const double *pPops,
                     double       *pDeltas) const
    {
        pDeltas[TUMOR] = tumorGrowthRate * pPops[TUMOR] - immuneKillTumorRate * pPops[TUMOR] * pPops[IMMUNE];
        pDeltas[IMMUNE] = -immuneDeathRate * pPops[IMMUNE] + immuneRecruitmentRate * pPops[TUMOR] * pPops[IMMUNE];
    }
};

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        TumorGrowth model = {std::sqrt(2.0), 0.5, 0.001, 0.001};

        std::vector<std::vector<double>> states = {{100.0, 10.0}};
        std::vector<double>              ts = {0.0};

        Hal::OdeSolver solver;
        solver.Runge45Record(
            [&](double t, const double *pState, double *pOut) { model.Derivatives(t, pState, pOut); },
            states, ts, 10.0, 0.001, 0.001, 0);

        std::vector<double> tumorX;
        std::vector<double> tumorY;
        std::vector<double> immuneX;
        std::vector<double> immuneY;
        for (size_t i = 0; i < states.size(); i++)
        {
            tumorX.push_back(ts[i]);
            tumorY.push_back(states[i][TUMOR]);
            immuneX.push_back(ts[i]);
            immuneY.push_back(states[i][IMMUNE]);
        }

        Hal::Examples::PlotWindow win("Tumor / Immune ODE", 250, 250, 2, !headless);
        win.AddLine(Hal::GREEN, tumorX, tumorY);
        win.AddLine(Hal::BLUE, immuneX, immuneY);
        win.Render(Hal::BLACK);
        win.ToPng("tumorimmunemodel.png");

        if (!headless)
        {
            win.Show();
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("TumorGrowth failed: ") + e.what());
        return 1;
    }
}
