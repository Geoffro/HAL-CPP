// A drug-treated tumor of sensitive (blue) and resistant (green) cells
// that the user treats interactively. A drug PDE diffuses in from the
// boundary while treatment is on; sensitive cells die in proportion to
// local drug, resistant ones do not. Each type's population is charted.
//
// Live (via common/InteractiveModel), press 1 to toggle the drug. Pass
// "headless" to run a scripted schedule (drug on after a warm-up) and dump
// the final frame to cliniciansim.png plus the final populations.

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Grids/PDEGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"
#include "common/InteractiveModel.h"
#include "common/TreatableTumor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

class ExampleModelInteractive;

class ExampleCellInteractive : public Hal::AgentSQ2DUnstackable<ExampleModelInteractive>
{
public:
    explicit ExampleCellInteractive(ExampleModelInteractive *pGrid)
        : Hal::AgentSQ2DUnstackable<ExampleModelInteractive>(pGrid),
          m_type(0)
    {
    }

    void CellStep();

    int32_t Type() const { return m_type; }
    void    SetType(int32_t type) { m_type = type; }

private:
    int32_t m_type;
};

class ExampleModelInteractive : public Hal::AgentGrid2D<ExampleCellInteractive, ExampleModelInteractive>,
                                public Hal::Examples::ITreatableTumor
{
public:
    static const int32_t RESISTANT;
    static const int32_t SENSITIVE;

    ExampleModelInteractive(int32_t xDim,
                            int32_t yDim)
        : Hal::AgentGrid2D<ExampleCellInteractive, ExampleModelInteractive>(xDim, yDim),
          m_drug(xDim, yDim),
          m_divHood(Hal::MooreHood(false)),
          m_pops(3, 0.0)
    {
    }

    ~ExampleModelInteractive()
    {
    }

    void InitTumor(int32_t radius,
                   double  resistantProb)
    {
        std::vector<int32_t> tumorHood = Hal::CircleHood(true, radius);
        int32_t hoodSize = MapHood(tumorHood.data(), static_cast<int32_t>(tumorHood.size()), Xdim() / 2, Ydim() / 2);
        for (int32_t i = 0; i < hoodSize; i++)
        {
            NewAgentSQ(tumorHood[i]).SetType((m_rng.Double() < resistantProb) ? RESISTANT : SENSITIVE);
        }
    }

    void ModelStep(bool drugOn)
    {
        ShuffleAgents(m_rng);
        for (ExampleCellInteractive &cell : *this)
        {
            cell.CellStep();
        }
        if (drugOn)
        {
            m_drug.DiffusionADI(DRUG_DIFF_RATE, DRUG_BOUNDARY_VAL);
        }
        else
        {
            m_drug.DiffusionADI(DRUG_DIFF_RATE);
        }
        m_drug.Update();
    }

    void InteractiveStep(const std::vector<double> &treatmentVals,
                         int32_t /*step*/) override
    {
        bool drugOn = !treatmentVals.empty() && (treatmentVals[0] != 0.0);
        for (int32_t i = 0; i < 10; i++)
        {
            ModelStep(drugOn);
        }
        double resistant = 0.0;
        double sensitive = 0.0;
        for (ExampleCellInteractive &c : *this)
        {
            if (c.Type() == RESISTANT)
            {
                resistant += 1.0;
            }
            else
            {
                sensitive += 1.0;
            }
        }
        m_pops[0] = resistant;
        m_pops[1] = sensitive;
        m_pops[2] = static_cast<double>(Pop());
    }

    void Draw(Hal::Examples::GridWindow &vis) override
    {
        for (int32_t i = 0; i < Length(); i++)
        {
            ExampleCellInteractive *pCell = GetAgent(i);
            vis.SetPix(i, pCell == nullptr ? Hal::HeatMapRgb(m_drug.Get(i)) : pCell->Type());
        }
    }

    std::vector<std::string> GetTreatmentNames() override { return {"Drug"}; }
    std::vector<int32_t>     GetTreatmentColors() override { return {Hal::Rgb256(238, 110, 46)}; }
    std::vector<int32_t>     GetPlotColors() override { return {RESISTANT, SENSITIVE, Hal::RED}; }
    std::vector<std::string> GetPlotLegendNames() override { return {"resistant", "sensitive", "total"}; }
    std::vector<double>      GetPlotVals() override { return m_pops; }

    Hal::PDEGrid2D &Drug() { return m_drug; }
    Hal::Rand      &Rng() { return m_rng; }
    int32_t        *DivHoodData() { return m_divHood.data(); }
    int32_t         DivHoodSize() const { return static_cast<int32_t>(m_divHood.size()); }

    static constexpr double DIV_PROB_SEN = 0.025;
    static constexpr double DIV_PROB_RES = 0.01;
    static constexpr double DEATH_PROB = 0.001;
    static constexpr double DRUG_DIFF_RATE = 2;
    static constexpr double DRUG_UPTAKE = -0.09;
    static constexpr double DRUG_DEATH = 0.2;
    static constexpr double DRUG_BOUNDARY_VAL = 1.0;

private:
    Hal::PDEGrid2D       m_drug;
    Hal::Rand            m_rng;
    std::vector<int32_t> m_divHood;
    std::vector<double>  m_pops;
};

const int32_t ExampleModelInteractive::RESISTANT = Hal::Rgb(0.0, 1.0, 0.0);
const int32_t ExampleModelInteractive::SENSITIVE = Hal::Rgb(0.0, 0.0, 1.0);

void ExampleCellInteractive::CellStep()
{
    ExampleModelInteractive *pGrid = this->Grid();
    pGrid->Drug().Mul(Isq(), ExampleModelInteractive::DRUG_UPTAKE);

    double drugDeath = (m_type == ExampleModelInteractive::RESISTANT)
                           ? 0.0
                           : pGrid->Drug().Get(Isq()) * ExampleModelInteractive::DRUG_DEATH;
    if (pGrid->Rng().Double() < (ExampleModelInteractive::DEATH_PROB + drugDeath))
    {
        Dispose();
        return;
    }

    double divProb = (m_type == ExampleModelInteractive::RESISTANT) ? ExampleModelInteractive::DIV_PROB_RES
                                                                  : ExampleModelInteractive::DIV_PROB_SEN;
    if (pGrid->Rng().Double() < divProb)
    {
        int32_t options = pGrid->MapEmptyHood(pGrid->DivHoodData(), pGrid->DivHoodSize(), Isq());
        if (options > 0)
        {
            pGrid->NewAgentSQ(pGrid->DivHoodData()[pGrid->Rng().Int(options)]).SetType(m_type);
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t x = 100;
        const int32_t y = 100;
        ExampleModelInteractive model(x, y);
        model.InitTumor(10, 0.5);

        Hal::Examples::InteractiveModel sim(model, x, y, 5, !headless);

        int32_t steps = headless ? 60 : 1000000;
        for (int32_t step = 0; (step < steps) && !sim.IsClosed(); step++)
        {
            if (headless)
            {
                // Scripted schedule: start the drug after a warm-up.
                sim.SetTreatment(0, (step > 20) ? 1.0 : 0.0);
            }
            else
            {
                sim.PollTreatmentKeys();
            }
            sim.Step(step);
            sim.Present(headless ? 0 : 20);
        }

        if (headless)
        {
            std::vector<double> pops = model.GetPlotVals();
            std::printf("final: resistant=%d sensitive=%d total=%d\n",
                        static_cast<int32_t>(pops[0]), static_cast<int32_t>(pops[1]), static_cast<int32_t>(pops[2]));
            sim.Vis().ToPng("cliniciansim.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("ClinicianSim failed: ") + e.what());
        return 1;
    }
}
