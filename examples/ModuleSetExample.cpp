// The division/death/mutation tumor decomposed into interchangeable
// behavior modules (setup, division/death probabilities, mutation
// bookkeeping, drawing) composed via a ModuleSet. Each module declares the
// hooks it implements when added, and the model invokes just those
// modules at each hook point. Per-module per-cell state uses VarSetManager
// and IVarSet.
//
// Live window by default; pass "headless" to dump the final tumor to
// modulesetexample.png.

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"
#include "Hal/Tools/ModuleSet.h"
#include "Hal/Tools/VarSetManager.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

class DivisionDeathMutation;
class CellEx;

// The fixed set of hook points modules can implement.
enum class Hook
{
    Setup,
    OnStep,
    OnStepCell,
    OnDiv,
    OnDeath,
    OnMut,
    SetDivProb,
    SetDeathProb,
    SetMutProb
};

// Module base: no-op / zero defaults, overridden by concrete modules. CellEx
// is only referenced by pointer here, so it may still be incomplete.
class DivDeathModule
{
public:
    explicit DivDeathModule(DivisionDeathMutation *pGrid)
        : m_pGrid(pGrid)
    {
    }
    virtual ~DivDeathModule() {}

    virtual void   Setup() {}
    virtual void   OnStep() {}
    virtual void   OnStepCell(CellEx *) {}
    virtual void   OnDiv(CellEx *, CellEx *) {}
    virtual void   OnDeath(CellEx *) {}
    virtual void   OnMut(CellEx *) {}
    virtual double SetDivProb(CellEx *, double) { return 0.0; }
    virtual double SetDeathProb(CellEx *, double) { return 0.0; }
    virtual double SetMutProb(CellEx *, double) { return 0.0; }

protected:
    DivisionDeathMutation *m_pGrid;
};

class CellEx : public Hal::AgentSQ2DUnstackable<DivisionDeathMutation>, public Hal::IVarSet
{
public:
    explicit CellEx(DivisionDeathMutation *pGrid)
        : Hal::AgentSQ2DUnstackable<DivisionDeathMutation>(pGrid),
          m_numMutations(0)
    {
    }

    void Mutate();

    const std::vector<double> &GetVars() const override { return m_vars; }
    void                       SetVars(std::vector<double> vars) override { m_vars = std::move(vars); }
    std::vector<double>       &Vars() { return m_vars; }

    int32_t NumMutations() const { return m_numMutations; }
    void    SetNumMutations(int32_t nMutations) { m_numMutations = nMutations; }

private:
    int32_t             m_numMutations;
    std::vector<double> m_vars;
};

class DivisionDeathMutation : public Hal::AgentGrid2D<CellEx, DivisionDeathMutation>
{
public:
    static constexpr int32_t MAX_MUTATIONS = 19;

    DivisionDeathMutation(int32_t x,
                          int32_t y)
        : Hal::AgentGrid2D<CellEx, DivisionDeathMutation>(x, y),
          m_hood(Hal::VonNeumannHood(false)),
          m_rng(1),
          m_divProbMods(0),
          m_dieProbMods(0),
          m_mutProbMods(0)
    {
    }

    void Setup()
    {
        for (DivDeathModule *pModule : m_mods.Iter(Hook::Setup))
        {
            pModule->Setup();
        }
        m_divProbMods = m_mods.CountModsWithMethod(Hook::SetDivProb);
        m_dieProbMods = m_mods.CountModsWithMethod(Hook::SetDeathProb);
        m_mutProbMods = m_mods.CountModsWithMethod(Hook::SetMutProb);
    }

    CellEx &NewCell(int32_t iSq,
                    CellEx *pParent)
    {
        CellEx &c = NewAgentSQ(iSq);
        m_vsm.AddVarSet(c);
        c.SetNumMutations((pParent == nullptr) ? 0 : pParent->NumMutations());
        for (DivDeathModule *pModule : m_mods.Iter(Hook::OnDiv))
        {
            pModule->OnDiv(pParent, &c);
        }
        return c;
    }

    void StepCells()
    {
        for (DivDeathModule *pModule : m_mods.Iter(Hook::OnStep))
        {
            pModule->OnStep();
        }
        int32_t hoodSize = HoodSize() / 3;
        for (CellEx &c : *this)
        {
            for (DivDeathModule *pModule : m_mods.Iter(Hook::OnStepCell))
            {
                pModule->OnStepCell(&c);
            }
            int32_t ct = MapEmptyHood(m_hood.data(), HoodSize(), c.Isq());
            // Integer division: 1 only when the neighborhood is completely full
            // (unused by these demo modules).
            double contactInhib = (hoodSize - ct) / hoodSize;

            double dieProb = 0.0;
            for (DivDeathModule *pModule : m_mods.Iter(Hook::SetDeathProb))
            {
                dieProb += pModule->SetDeathProb(&c, contactInhib);
            }
            dieProb /= m_dieProbMods;
            if ((dieProb > 0.0) && (m_rng.Double() < dieProb))
            {
                for (DivDeathModule *pModule : m_mods.Iter(Hook::OnDeath))
                {
                    pModule->OnDeath(&c);
                }
                c.Dispose();
                continue;
            }

            double mutProb = 0.0;
            if (c.NumMutations() < MAX_MUTATIONS)
            {
                for (DivDeathModule *pModule : m_mods.Iter(Hook::SetMutProb))
                {
                    mutProb += pModule->SetMutProb(&c, contactInhib);
                }
            }
            mutProb /= m_mutProbMods;
            if ((mutProb > 0.0) && (m_rng.Double() < mutProb))
            {
                c.Mutate();
            }

            if (ct > 0)
            {
                double divProb = 0.0;
                for (DivDeathModule *pModule : m_mods.Iter(Hook::SetDivProb))
                {
                    divProb += pModule->SetDivProb(&c, contactInhib);
                }
                divProb /= m_divProbMods;
                if ((divProb > 0.0) && (m_rng.Double() < divProb))
                {
                    NewCell(m_hood[m_rng.Int(ct)], &c);
                }
            }
        }
        ShuffleAgents(m_rng);
        IncTick();
    }

    Hal::ModuleSet<DivDeathModule, Hook> &Mods() { return m_mods; }
    Hal::VarSetManager                   &Vsm() { return m_vsm; }
    Hal::Rand                            &Rng() { return m_rng; }
    int32_t                              *HoodData() { return m_hood.data(); }
    int32_t                               HoodSize() const { return static_cast<int32_t>(m_hood.size()); }

private:
    Hal::ModuleSet<DivDeathModule, Hook> m_mods;
    Hal::VarSetManager                   m_vsm;
    std::vector<int32_t>                 m_hood;
    Hal::Rand                            m_rng;
    int32_t                              m_divProbMods;
    int32_t                              m_dieProbMods;
    int32_t                              m_mutProbMods;
};

void CellEx::Mutate()
{
    m_numMutations++;
    for (DivDeathModule *pModule : this->Grid()->Mods().Iter(Hook::OnMut))
    {
        pModule->OnMut(this);
    }
}

// --- Concrete modules ---

class SetupTumorMod : public DivDeathModule
{
public:
    SetupTumorMod(DivisionDeathMutation *pGrid,
                  double                 radius)
        : DivDeathModule(pGrid),
          m_radius(radius)
    {
    }

    void Setup() override
    {
        std::vector<int32_t> circleHood = Hal::CircleHood(true, m_radius);
        int32_t len = m_pGrid->MapHood(circleHood.data(), static_cast<int32_t>(circleHood.size()), m_pGrid->Xdim() / 2, m_pGrid->Ydim() / 2);
        for (int32_t i = 0; i < len; i++)
        {
            m_pGrid->NewCell(circleHood[i], nullptr);
        }
    }

private:
    double m_radius;
};

class DivDeathProbMod : public DivDeathModule
{
public:
    explicit DivDeathProbMod(DivisionDeathMutation *pGrid)
        : DivDeathModule(pGrid)
    {
    }

    double SetDivProb(CellEx *, double) override { return 0.2; }
    double SetDeathProb(CellEx *, double) override { return 0.1; }
};

class DivMutMod : public DivDeathModule
{
public:
    explicit DivMutMod(DivisionDeathMutation *pGrid)
        : DivDeathModule(pGrid),
          DIVIDED(pGrid->Vsm().NewVar())
    {
    }

    void OnDiv(CellEx *pParent, CellEx *pCell) override
    {
        if (pParent != nullptr)
        {
            pParent->Vars()[DIVIDED] = 1;
        }
        pCell->Vars()[DIVIDED] = 1;
    }

    double SetMutProb(CellEx *pCell, double) override
    {
        if (pCell->Vars()[DIVIDED] == 1)
        {
            pCell->Vars()[DIVIDED] = 0;
            return 0.1;
        }
        return 0.0;
    }

private:
    int32_t DIVIDED;
};

class DrawTumorMod : public DivDeathModule
{
public:
    DrawTumorMod(DivisionDeathMutation      *pGrid,
                 Hal::Examples::GridWindow  &vis)
        : DivDeathModule(pGrid),
          m_vis(vis)
    {
    }

    void OnDiv(CellEx *, CellEx *pC) override { Draw(pC); }
    void OnMut(CellEx *pC) override { Draw(pC); }
    void OnDeath(CellEx *pC) override { m_vis.SetPix(pC->Isq(), Hal::BLACK); }

private:
    void Draw(CellEx *pC)
    {
        m_vis.SetPix(pC->Isq(), Hal::CategoricalColor(pC->NumMutations()));
    }

    Hal::Examples::GridWindow &m_vis;
};

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t x = 250;
        const int32_t y = 250;
        Hal::Examples::GridWindow vis("ModuleSet tumor", x, y, 4, !headless);
        DivisionDeathMutation grid(x, y);

        grid.Mods().AddModule(std::make_unique<SetupTumorMod>(&grid, 10), {Hook::Setup});
        grid.Mods().AddModule(std::make_unique<DivDeathProbMod>(&grid), {Hook::SetDivProb, Hook::SetDeathProb});
        grid.Mods().AddModule(std::make_unique<DrawTumorMod>(&grid, vis), {Hook::OnDiv, Hook::OnMut, Hook::OnDeath});
        grid.Mods().AddModule(std::make_unique<DivMutMod>(&grid), {Hook::OnDiv, Hook::SetMutProb});
        grid.Setup();

        for (int32_t tick = 0; (tick < 1000) && !vis.IsClosed(); tick++)
        {
            grid.StepCells();
            vis.TickPause(0);
        }

        if (headless)
        {
            vis.ToPng("modulesetexample.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("ModuleSetExample failed: ") + e.what());
        return 1;
    }
}
