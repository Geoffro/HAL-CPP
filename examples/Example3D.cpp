// A 3D tumor growing through an oxygen field. Blood vessels (large red
// spheres in vertical columns) are oxygen sources; tumor cells consume
// oxygen, divide where it is plentiful, and die where it is scarce. Oxygen
// diffuses on a PDEGrid3D. Rendered with the software Scene3D, tumor cells
// colored by local oxygen.
//
// Live window by default (drag to orbit, up/down to zoom); pass "headless"
// to run a bounded number of steps and dump the final frame to example3d.png.

#include "Hal/Agents/AgentSQ3DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid3D.h"
#include "Hal/Grids/Grid2D.h"
#include "Hal/Grids/PDEGrid3D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"
#include "common/Scene3D.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <numeric>
#include <string>
#include <vector>

class Example3D;

class ExCell3D : public Hal::AgentSQ3DUnstackable<Example3D>
{
public:
    static constexpr int32_t VESSEL = 0;
    static constexpr int32_t TUMOR = 1;

    explicit ExCell3D(Example3D *pGrid)
        : Hal::AgentSQ3DUnstackable<Example3D>(pGrid),
          m_type(TUMOR)
    {
    }

    void InitVessel();
    void InitTumor() { m_type = TUMOR; }
    void Metabolism();
    bool Death();
    bool Divide();
    void CellStep();

    int32_t Type() const { return m_type; }

private:
    int32_t m_type;
};

class Example3D : public Hal::AgentGrid3D<ExCell3D, Example3D>
{
public:
    static const int32_t VESSEL_COLOR;

    Example3D(int32_t x,
              int32_t y,
              int32_t z)
        : Hal::AgentGrid3D<ExCell3D, Example3D>(x, y, z),
          m_oxygen(x, y, z),
          m_vnHood(Hal::VonNeumannHood3D(false)),
          // Custom hood: lateral moves in x and z (padded 3D layout).
          m_vnHood2D{0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, -1}
    {
    }

    void DiffStep()
    {
        for (ExCell3D &cell : *this)
        {
            cell.Metabolism();
        }
        m_oxygen.MulAll(NORMAL_METABOLISM_RATE);
        m_oxygen.Diffusion(DIFF_RATE);
        m_oxygen.Update();
    }

    void StepAll()
    {
        m_countTumor = 0;
        for (ExCell3D &cell : *this)
        {
            cell.CellStep();
        }
        if (m_countTumor == 0)
        {
            int32_t placeLoc = m_rng.Int(Length());
            if (GetAgent(placeLoc) == nullptr)
            {
                NewAgentSQ(placeLoc).InitTumor();
            }
        }
        DiffStep();
    }

    void GenVessels(double vesselSpacingMin,
                    double migProb)
    {
        Hal::Grid2D<double>  openSpots(Xdim(), Zdim());
        std::vector<int32_t> spacingHood = Hal::CircleHood(false, vesselSpacingMin);
        std::vector<int32_t> indices(openSpots.Length());
        std::iota(indices.begin(), indices.end(), 0);
        m_rng.Shuffle(indices.data(), static_cast<int32_t>(indices.size()));

        for (int32_t i : indices)
        {
            if (openSpots.Get(i) == 0)
            {
                int32_t x = openSpots.ItoX(i);
                int32_t z = openSpots.ItoY(i);
                GenVessel(x, z, migProb);
                int32_t nSpots = openSpots.MapHood(spacingHood.data(), static_cast<int32_t>(spacingHood.size()), x, z);
                for (int32_t j = 0; j < nSpots; j++)
                {
                    openSpots.Set(spacingHood[j], -1.0);
                }
            }
        }
    }

    void DrawCells(Hal::Examples::Scene3D &scene)
    {
        scene.ClearBox(Hal::Rgb256(38, 1, 5), Hal::Rgb(1, 0, 0));
        for (ExCell3D &cell : *this)
        {
            if (cell.Type() == ExCell3D::VESSEL)
            {
                scene.Sphere(cell.Xpt(), cell.Ypt(), cell.Zpt(), 1.0, VESSEL_COLOR);
            }
            else
            {
                double o = m_oxygen.Get(cell.Isq());
                scene.Sphere(cell.Xpt(), cell.Ypt(), cell.Zpt(), 0.3, Hal::HeatMapBrg(std::pow(o, 0.5) * 0.8 + 0.2));
            }
        }
        scene.Render();
    }

    void AddVessel(ExCell3D *pCell) { m_vessels.push_back(pCell); }
    void IncTumorCount() { m_countTumor++; }

    Hal::PDEGrid3D &Oxygen() { return m_oxygen; }
    Hal::Rand      &Rng() { return m_rng; }
    int32_t        *VnHoodData() { return m_vnHood.data(); }
    int32_t         VnHoodSize() const { return static_cast<int32_t>(m_vnHood.size()); }

    static constexpr double DIFF_RATE = 0.5 / 6;
    static constexpr double TUMOR_METABOLISM_RATE = -0.04;
    static constexpr double NORMAL_METABOLISM_RATE = -0.005;
    static constexpr double VESSEL_CONC = 1;
    static constexpr double DEATH_CONC = 0.01;

private:
    void GenVessel(int32_t x,
                   int32_t z,
                   double  migProb)
    {
        for (int32_t y = 0; y < Ydim(); y++)
        {
            if (m_rng.Double() < migProb)
            {
                int32_t openCt = MapHood(m_vnHood2D.data(), static_cast<int32_t>(m_vnHood2D.size()), x, y, z);
                int32_t i = m_vnHood2D[m_rng.Int(openCt)];
                x = ItoX(i);
                z = ItoZ(i);
            }
            ExCell3D *pOccupant = GetAgent(x, y, z);
            if (pOccupant != nullptr)
            {
                pOccupant->Dispose();
            }
            NewAgentSQ(x, y, z).InitVessel();
        }
    }

    Hal::PDEGrid3D           m_oxygen;
    Hal::Rand                m_rng;
    std::vector<int32_t>     m_vnHood;
    std::vector<int32_t>     m_vnHood2D;
    std::vector<ExCell3D *>  m_vessels;
    int32_t                  m_countTumor = 0;
};

const int32_t Example3D::VESSEL_COLOR = Hal::Rgb256(255, 78, 68);

void ExCell3D::InitVessel()
{
    m_type = VESSEL;
    this->Grid()->AddVessel(this);
}

void ExCell3D::Metabolism()
{
    Example3D *pGrid = this->Grid();
    switch (m_type)
    {
        case VESSEL:
            pGrid->Oxygen().Set(Isq(), Example3D::VESSEL_CONC);
            break;
        case TUMOR:
            pGrid->Oxygen().Mul(Isq(), Example3D::TUMOR_METABOLISM_RATE);
            break;
        default:
            break;
    }
}

bool ExCell3D::Death()
{
    Example3D *pGrid = this->Grid();
    double     resVal = pGrid->Oxygen().Get(Isq());
    return (resVal < Example3D::DEATH_CONC) && (pGrid->Rng().Double() < (1.0 - resVal / Example3D::DEATH_CONC));
}

bool ExCell3D::Divide()
{
    Example3D *pGrid = this->Grid();
    return pGrid->Rng().Double() < pGrid->Oxygen().Get(Isq());
}

void ExCell3D::CellStep()
{
    Example3D *pGrid = this->Grid();
    if (m_type == TUMOR)
    {
        pGrid->IncTumorCount();
        if (Death())
        {
            Dispose();
            return;
        }
        if (Divide())
        {
            int32_t nDivOpts = pGrid->MapEmptyHood(pGrid->VnHoodData(), pGrid->VnHoodSize(), Isq());
            if (nDivOpts > 1)
            {
                pGrid->NewAgentSQ(pGrid->VnHoodData()[pGrid->Rng().Int(nDivOpts)]).InitTumor();
            }
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        // An 80 x 20 x 80 domain scaled down so the software renderer and the
        // headless run stay quick.
        const int32_t x = headless ? 40 : 80;
        const int32_t yDim = headless ? 10 : 20;
        const int32_t z = headless ? 40 : 80;

        Example3D ex(x, yDim, z);
        ex.GenVessels(15, 0.8);
        for (int32_t i = 0; i < (headless ? 40 : 100); i++)
        {
            ex.DiffStep();
        }

        Hal::Examples::GridWindow win("Tumor Vis", 800, 800, 1, !headless);
        Hal::Examples::Scene3D    scene(win, x, yDim, z);

        int32_t steps = headless ? 60 : 1000000;
        for (int32_t i = 0; (i < steps) && !win.IsClosed(); i++)
        {
            scene.HandleInput();
            ex.StepAll();
            ex.DrawCells(scene);
            ex.CleanAgents();
            ex.ShuffleAgents(ex.Rng());
        }

        if (headless)
        {
            win.ToPng("example3d.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("Example3D failed: ") + e.what());
        return 1;
    }
}
