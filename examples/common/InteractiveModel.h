#pragma once

#include "GridWindow.h"
#include "PlotWindow.h"
#include "TreatableTumor.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    namespace Examples
    {
        // A reduced software port of Java's Tools/InteractiveModel — the
        // "clinician simulator" driver. It shows an ITreatableTumor model in
        // one window and its reported plot values over time in a second
        // window, while the user toggles treatments on/off with the number
        // keys (1..N for treatment 1..N). The full Java version is a
        // mouse-driven treatment-timeline editor with scrubbing and scoring;
        // this keeps the interactive-treatment core without that UI.
        //
        // Each Step advances the model under the current treatment intensities
        // and appends the plot values to the history charted by RefreshPlot.
        class InteractiveModel
        {
        public:
            InteractiveModel(ITreatableTumor &model,
                             int32_t          visX,
                             int32_t          visY,
                             int32_t          visScale,
                             bool             active);

            ~InteractiveModel();

            // Toggle treatment intensities from the model window's number keys
            // (live only; no-op headless). Call once per frame before Step.
            void PollTreatmentKeys();

            // Set treatment i's intensity directly (for scripted/headless use).
            void SetTreatment(int32_t treatment,
                              double  intensity);

            // Advance the model one step and record its plot values.
            void Step(int32_t step);

            // Redraw the population plot from the recorded history.
            void RefreshPlot();

            // Present both windows, pacing to pauseMillis.
            void Present(int32_t pauseMillis);

            bool IsClosed() const;

            GridWindow &Vis();

        private:
            ITreatableTumor                 &m_model;
            GridWindow                       m_vis;
            PlotWindow                       m_plot;
            std::vector<double>              m_treatmentVals;
            std::vector<std::vector<double>> m_plotHistory;
            std::vector<double>              m_timeAxis;
            std::vector<bool>                m_keyWasDown;
        };
    }
}
