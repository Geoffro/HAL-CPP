#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Hal
{
    namespace Examples
    {
        class GridWindow;

        // The model contract driven by InteractiveModel: a clinician toggles one
        // or more treatments over time; each step the model advances under the
        // current treatment intensities, draws itself, and reports plot values
        // (e.g. subpopulation counts) to chart.
        class ITreatableTumor
        {
        public:
            ITreatableTumor()
            {
            }
            virtual ~ITreatableTumor()
            {
            }

            // Advance the model by one interactive step under the given
            // per-treatment intensities (index i is treatment i's intensity),
            // updating whatever GetPlotVals will report.
            virtual void InteractiveStep(const std::vector<double> &treatmentVals,
                                         int32_t                    step) = 0;

            // Render the current model state into the visualization window.
            virtual void Draw(GridWindow &vis) = 0;

            // One name / display color per treatment.
            virtual std::vector<std::string> GetTreatmentNames() = 0;
            virtual std::vector<int32_t>     GetTreatmentColors() = 0;

            // One color / legend name / value per plotted series.
            virtual std::vector<int32_t>     GetPlotColors() = 0;
            virtual std::vector<std::string> GetPlotLegendNames() = 0;
            virtual std::vector<double>      GetPlotVals() = 0;
        };
    }
}
