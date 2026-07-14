// Ported from Examples/_6CompetitiveRelease: three side-by-side copies of a
// drug-treated tumor of sensitive (blue) and resistant (green) cells,
// differing only in dosing schedule (none / constant / periodic). A drug
// PDE diffuses in from the boundary during dosing; sensitive cells die in
// proportion to local drug, illustrating competitive release of the
// resistant clone under aggressive dosing.
//
// Live window by default; pass "headless" to run fewer steps and dump the
// final three-panel state to competitiverelease.png (and populations.csv).

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Grids/PDEGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/MathUtil.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

class ExampleModel;

class ExampleCell : public Hal::AgentSQ2DUnstackable<ExampleModel>
{
public:
    explicit ExampleCell(ExampleModel *pGrid)
        : Hal::AgentSQ2DUnstackable<ExampleModel>(pGrid),
          m_type(0)
    {
    }

    void CellStep();

    int32_t Type() const { return m_type; }
    void    SetType(int32_t type) { m_type = type; }

private:
    int32_t m_type;
};

class ExampleModel : public Hal::AgentGrid2D<ExampleCell, ExampleModel>
{
public:
    static const int32_t RESISTANT;
    static const int32_t SENSITIVE;

    ExampleModel(int32_t   xDim,
                 int32_t   yDim,
                 uint64_t  seed)
        : Hal::AgentGrid2D<ExampleCell, ExampleModel>(xDim, yDim),
          m_drug(xDim, yDim),
          m_rng(seed),
          m_divHood(Hal::MooreHood(false))
    {
    }

    void InitTumor(double radius,
                   double resistantProb)
    {
        std::vector<int32_t> tumorHood = Hal::CircleHood(true, radius);
        int32_t hoodSize = MapHood(tumorHood.data(), static_cast<int32_t>(tumorHood.size()), Xdim() / 2, Ydim() / 2);
        for (int32_t i = 0; i < hoodSize; i++)
        {
            NewAgentSQ(tumorHood[i]).SetType((m_rng.Double() < resistantProb) ? RESISTANT : SENSITIVE);
        }
    }

    void ModelStep(int32_t tick)
    {
        ShuffleAgents(m_rng);
        for (ExampleCell &cell : *this)
        {
            cell.CellStep();
        }
        double periodTick = std::fmod(tick - DRUG_START, DRUG_PERIOD);
        if ((periodTick > 0) && (periodTick < DRUG_DURATION))
        {
            m_drug.DiffusionADI(DRUG_DIFF_RATE, DRUG_BOUNDARY_VAL);
        }
        else
        {
            m_drug.DiffusionADI(DRUG_DIFF_RATE);
        }
        m_drug.Update();
    }

    void DrawModel(Hal::Examples::GridWindow &vis,
                   int32_t                    iModel)
    {
        for (int32_t x = 0; x < Xdim(); x++)
        {
            for (int32_t y = 0; y < Ydim(); y++)
            {
                ExampleCell *pDrawMe = GetAgent(x, y);
                if (pDrawMe != nullptr)
                {
                    vis.SetPix(x + iModel * Xdim(), y, pDrawMe->Type());
                }
                else
                {
                    vis.SetPix(x + iModel * Xdim(), y, Hal::HeatMapRgb(m_drug.Get(x, y)));
                }
            }
        }
    }

    Hal::PDEGrid2D &Drug() { return m_drug; }
    Hal::Rand      &Rng() { return m_rng; }
    int32_t        *DivHoodData() { return m_divHood.data(); }
    int32_t         DivHoodSize() const { return static_cast<int32_t>(m_divHood.size()); }

    // Time-scaled rates (2 hours per timestep), matching the Java model.
    static constexpr double TIMESTEP = 2.0 / 24;
    static constexpr double SPACE_STEP = 20;
    const double            DIV_PROB_SEN = Hal::ProbScale(0.5, TIMESTEP);
    const double            DIV_PROB_RES = Hal::ProbScale(0.2, TIMESTEP);
    const double            DEATH_PROB = Hal::ProbScale(0.02, TIMESTEP);
    const double            DRUG_DEATH = Hal::ProbScale(0.8, TIMESTEP);
    const double            DRUG_START = 20 / TIMESTEP;
    const double            DRUG_PERIOD = 15 / TIMESTEP;
    double                  DRUG_DURATION = 2 / TIMESTEP;
    const double            DRUG_UPTAKE = -0.03 * TIMESTEP;
    const double            DRUG_DIFF_RATE = 0.02 * 60 * 60 * 24 * (TIMESTEP / (SPACE_STEP * SPACE_STEP));
    const double            DRUG_BOUNDARY_VAL = 1.0;

private:
    Hal::PDEGrid2D       m_drug;
    Hal::Rand            m_rng;
    std::vector<int32_t> m_divHood;
};

const int32_t ExampleModel::RESISTANT = Hal::Rgb(0.0, 1.0, 0.0);
const int32_t ExampleModel::SENSITIVE = Hal::Rgb(0.0, 0.0, 1.0);

void ExampleCell::CellStep()
{
    ExampleModel *pGrid = this->Grid();
    pGrid->Drug().Mul(Isq(), pGrid->DRUG_UPTAKE);

    double deathProb;
    if (m_type == ExampleModel::RESISTANT)
    {
        deathProb = pGrid->DEATH_PROB;
    }
    else
    {
        deathProb = pGrid->DEATH_PROB + pGrid->Drug().Get(Isq()) * pGrid->DRUG_DEATH;
    }
    if (pGrid->Rng().Double() < deathProb)
    {
        Dispose();
        return;
    }

    double divProb = (m_type == ExampleModel::RESISTANT) ? pGrid->DIV_PROB_RES : pGrid->DIV_PROB_SEN;
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
        const int32_t visScale = headless ? 2 : 5;
        const int32_t tumorRad = 10;
        const double  resistantProb = 0.5;

        Hal::Examples::GridWindow win("Competitive Release", x * 3, y, visScale, !headless);
        std::ofstream popsOut("populations.csv");

        std::vector<std::unique_ptr<ExampleModel>> models;
        for (int32_t i = 0; i < 3; i++)
        {
            models.push_back(std::make_unique<ExampleModel>(x, y, 0));
            models.back()->InitTumor(tumorRad, resistantProb);
        }
        models[0]->DRUG_DURATION = 0;   // no drug
        models[1]->DRUG_DURATION = 200; // constant drug
        // models[2] keeps the default periodic schedule.

        int32_t ticks = headless ? 600 : 10000;
        for (int32_t tick = 0; (tick < ticks) && !win.IsClosed(); tick++)
        {
            win.TickPause(headless ? 0 : 5);
            for (int32_t i = 0; i < 3; i++)
            {
                models[i]->ModelStep(tick);
                models[i]->DrawModel(win, i);
            }
            popsOut << models[0]->Pop() << "," << models[1]->Pop() << "," << models[2]->Pop() << "\n";
        }
        popsOut.close();

        if (headless)
        {
            win.ToPng("competitiverelease.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("CompetitiveRelease failed: ") + e.what());
        return 1;
    }
}
