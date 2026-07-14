// Ported from Examples/PhylogenyTracker/PhylogenyModel: a growing tumor
// whose cells carry a Genome tracked in a PhylogenyTracker. On mutation a
// cell forks a new child genome (recoloring slightly), so the clonal
// structure is recorded as a tree. Cells are drawn by their genome color,
// producing the characteristic clonal-sweep patches.
//
// Genomes are created via tracker.NewGenome(pParent, ...) (not `new`, as in
// Java) since the C++ tracker owns them. The run records clone sizes each
// tick and writes them to clones.csv (a Muller-plot input); a live window
// shows the clones, or pass "headless" to also dump phylogenymodel.png.

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"
#include "Hal/Tools/PhylogenyTracker.h"

#include "common/GridWindow.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

class CellGenome : public Hal::Genome<CellGenome>
{
public:
    CellGenome(CellGenome *pParent,
               Hal::Rand  &rn)
        : m_numMutations(0),
          m_color(0)
    {
        if (pParent != nullptr)
        {
            m_numMutations = pParent->m_numMutations + 1;
            m_color = pParent->m_color;
            MutateColor(rn, 40);
        }
        else
        {
            m_numMutations = 0;
            m_color = Hal::Rgb256(128, 128, 128);
        }
    }

    int32_t GenMutComp(int32_t initialVal,
                       Hal::Rand &rn,
                       int32_t mutRate)
    {
        return initialVal + rn.Int(mutRate * 2 + 1) - mutRate;
    }

    void MutateColor(Hal::Rand &rn,
                     int32_t mutRate)
    {
        int32_t newR = GenMutComp(Hal::GetRed256(m_color), rn, mutRate);
        int32_t newG = GenMutComp(Hal::GetGreen256(m_color), rn, mutRate);
        int32_t newB = GenMutComp(Hal::GetBlue256(m_color), rn, mutRate);
        m_color = Hal::Rgb256(newR, newG, newB);
    }

    int32_t NumMutations() const { return m_numMutations; }
    int32_t Color() const { return m_color; }

private:
    int32_t m_numMutations;
    int32_t m_color;
};

class PhylogenyModel;

class CellEx : public Hal::AgentSQ2DUnstackable<PhylogenyModel>
{
public:
    explicit CellEx(PhylogenyModel *pGrid)
        : Hal::AgentSQ2DUnstackable<PhylogenyModel>(pGrid),
          m_pMyGenome(nullptr)
    {
    }

    void PossiblyMutate();
    void Draw();
    void Divide();

    CellGenome *Genome() const { return m_pMyGenome; }
    void        SetGenome(CellGenome *pGenome) { m_pMyGenome = pGenome; }

private:
    CellGenome *m_pMyGenome;
};

class PhylogenyModel : public Hal::AgentGrid2D<CellEx, PhylogenyModel>
{
public:
    PhylogenyModel(int32_t                     x,
                   int32_t                     y,
                   Hal::Examples::GridWindow  &vis)
        : Hal::AgentGrid2D<CellEx, PhylogenyModel>(x, y),
          m_hood(Hal::VonNeumannHood(false)),
          m_tracker(false),
          m_vis(vis)
    {
        m_pSeed = &m_tracker.NewGenome(nullptr, m_rng);
    }

    void InitTumor(double radius)
    {
        std::vector<int32_t> circleHood = Hal::CircleHood(true, radius);
        int32_t len = MapHood(circleHood.data(), static_cast<int32_t>(circleHood.size()), Xdim() / 2, Ydim() / 2);
        for (int32_t i = 0; i < len; i++)
        {
            CellEx &c = NewAgentSQ(circleHood[i]);
            c.SetGenome(&m_tracker.NewGenome(m_pSeed, m_rng));
            c.Genome()->IncPop();
            c.Draw();
        }
    }

    void StepCells()
    {
        for (CellEx &c : *this)
        {
            if (m_rng.Double() < DIE_PROB)
            {
                m_vis.SetPix(c.Isq(), Hal::BLACK);
                c.Genome()->DecPop();
                c.Dispose();
            }
            else if (m_rng.Double() < (DIV_PROB * std::pow(MUT_ADVANTAGE, c.Genome()->NumMutations())))
            {
                c.Divide();
            }
        }
        ShuffleAgents(m_rng);
    }

    Hal::PhylogenyTracker<CellGenome> &Tracker() { return m_tracker; }
    Hal::Examples::GridWindow         &Vis() { return m_vis; }
    Hal::Rand                         &Rng() { return m_rng; }
    int32_t                           *HoodData() { return m_hood.data(); }
    int32_t                            HoodSize() const { return static_cast<int32_t>(m_hood.size()); }

    static constexpr double DIV_PROB = 0.2;
    static constexpr double MUT_PROB = 0.001;
    static constexpr double DIE_PROB = 0.1;
    static constexpr double MUT_ADVANTAGE = 1.08;
    static constexpr int32_t MAX_MUTATIONS = 19;

private:
    std::vector<int32_t>              m_hood;
    Hal::PhylogenyTracker<CellGenome> m_tracker;
    CellGenome                       *m_pSeed;
    Hal::Rand                         m_rng;
    Hal::Examples::GridWindow        &m_vis;
};

void CellEx::PossiblyMutate()
{
    PhylogenyModel *pGrid = this->Grid();
    if ((m_pMyGenome->NumMutations() < PhylogenyModel::MAX_MUTATIONS) && (pGrid->Rng().Double() < PhylogenyModel::MUT_PROB))
    {
        m_pMyGenome->DecPop();
        m_pMyGenome = &pGrid->Tracker().NewGenome(m_pMyGenome, pGrid->Rng());
        m_pMyGenome->IncPop();
        Draw();
    }
}

void CellEx::Draw()
{
    this->Grid()->Vis().SetPix(Isq(), m_pMyGenome->Color());
}

void CellEx::Divide()
{
    PhylogenyModel *pGrid = this->Grid();
    int32_t         nOpts = pGrid->MapEmptyHood(pGrid->HoodData(), pGrid->HoodSize(), Isq());
    if (nOpts > 0)
    {
        int32_t iDaughter = pGrid->HoodData()[pGrid->Rng().Int(nOpts)];
        CellEx &daughter = pGrid->NewAgentSQ(iDaughter);
        daughter.SetGenome(m_pMyGenome);
        m_pMyGenome->IncPop();
        daughter.Draw();
        PossiblyMutate();
        daughter.PossiblyMutate();
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t x = 500;
        const int32_t y = 500;
        Hal::Examples::GridWindow vis("Phylogeny (clonal sweeps)", x, y, headless ? 1 : 2, !headless);
        vis.Clear(Hal::BLACK);
        PhylogenyModel grid(x, y, vis);
        grid.InitTumor(10);

        for (int32_t tick = 0; (tick < 1000) && !vis.IsClosed(); tick++)
        {
            vis.TickPause(headless ? 0 : 10);
            grid.StepCells();
            grid.Tracker().RecordClones(std::to_string(tick));
        }

        grid.Tracker().OutputClonesToCSV(
            "clones.csv",
            {"RGB"},
            [](CellGenome &g) {
                return std::to_string(Hal::GetRed256(g.Color())) + " " +
                       std::to_string(Hal::GetGreen256(g.Color())) + " " +
                       std::to_string(Hal::GetBlue256(g.Color()));
            },
            0);

        if (headless)
        {
            vis.ToPng("phylogenymodel.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("PhylogenyModel failed: ") + e.what());
        return 1;
    }
}
