// Ported from Examples/PopulationTumor/PopTumor: a count-based (mean-field)
// tumor with four subpopulations per lattice site — wild-type plus two
// single mutants (M1, M2) and the double mutant (MBOTH) — each a
// PopulationGrid2D. Every tick each population divides (density-limited by
// the total local count), dies, mutates a fraction of its divisions up the
// mutation lattice, and migrates by multinomial diffusion. Four time
// snapshots are drawn side by side, colored by subpopulation mix.
//
// Uses PopulationGrid2D::Diffusion (mean-field multinomial diffusion) and
// ApplyOccupied. Live window by default; pass "headless" to dump the final
// snapshot strip to poptumor.png.

#include "Hal/Color.h"
#include "Hal/Grids/PopulationGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"
#include "Hal/Tools/MultinomialCalc.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

static const int32_t TUMOR = 0;
static const int32_t M1 = 1;
static const int32_t M2 = 2;
static const int32_t MBOTH = 3;
static const int32_t N_TYPES = 4;

class PopTumor
{
public:
    PopTumor(int32_t                    sideLen,
             double                     divRate,
             double                     deathRate,
             double                     migrationRate,
             int32_t                    latticeCap,
             double                     mutProb1,
             double                     mutProb2,
             Hal::Examples::GridWindow &win)
        : m_sideLen(sideLen),
          m_divRate(divRate),
          m_divRateMut(divRate + 0.01),
          m_divRateDoubleMut(divRate + 0.02),
          m_deathRate(deathRate),
          m_migrationRate(migrationRate),
          m_latticeCap(latticeCap),
          m_mutProb1(mutProb1),
          m_mutProb2(mutProb2),
          m_cellMn(m_rng),
          m_sums(static_cast<size_t>(sideLen) * sideLen, 0),
          m_win(win)
    {
        for (int32_t i = 0; i < N_TYPES; i++)
        {
            m_cells.push_back(std::make_unique<Hal::PopulationGrid2D<int32_t>>(sideLen, sideLen, false, false));
        }
    }

    void Init(int32_t initPop)
    {
        m_cells[TUMOR]->Add(m_sideLen / 2, m_sideLen / 2, initPop);
        m_cells[TUMOR]->Update();
    }

    int64_t TotalPop() const
    {
        int64_t pop = 0;
        for (const auto &g : m_cells)
        {
            pop += g->Pop();
        }
        return pop;
    }

    void Step(int32_t idraw)
    {
        for (int32_t type = 0; type < N_TYPES; type++)
        {
            Hal::PopulationGrid2D<int32_t> &grid = *m_cells[type];
            grid.ApplyOccupied([&](int32_t i, int32_t ct) {
                m_cellMn.Setup(ct);
                double  density = 1.0 - m_sums[i] * 1.0 / m_latticeCap;
                int32_t ctDiv;
                if ((type == M1) || (type == M2))
                {
                    ctDiv = m_cellMn.Sample(m_divRateMut * density);
                }
                else if (type == MBOTH)
                {
                    ctDiv = m_cellMn.Sample(m_divRateDoubleMut * density);
                }
                else
                {
                    ctDiv = m_cellMn.Sample(m_divRate * density);
                }
                int32_t ctDie = m_cellMn.Sample(m_deathRate);
                int32_t ctM1mut = 0;
                int32_t ctM2mut = 0;
                int32_t ctBothmut = 0;
                if (type == TUMOR)
                {
                    m_cellMn.Setup(ctDiv);
                    ctM1mut = m_cellMn.Sample(m_mutProb1);
                    ctM2mut = m_cellMn.Sample(m_mutProb2);
                }
                else if (type == M1)
                {
                    ctBothmut = m_rng.Binomial(ctDiv, m_mutProb2);
                }
                else if (type == M2)
                {
                    ctBothmut = m_rng.Binomial(ctDiv, m_mutProb1);
                }
                grid.Add(i, ctDiv - (ctDie + ctM1mut + ctM2mut + ctBothmut));
                if (ctM1mut > 0)
                {
                    m_cells[M1]->Add(i, ctM1mut);
                }
                if (ctM2mut > 0)
                {
                    m_cells[M2]->Add(i, ctM2mut);
                }
                if (ctBothmut > 0)
                {
                    m_cells[MBOTH]->Add(i, ctBothmut);
                }
            });
        }
        for (auto &g : m_cells)
        {
            g->Update();
        }
        for (auto &g : m_cells)
        {
            g->Diffusion(m_migrationRate, m_cellMn);
            g->Update();
        }

        m_cells[TUMOR]->CopyTo(m_sums.data());
        for (int32_t i = 1; i < N_TYPES; i++)
        {
            m_cells[i]->AddTo(m_sums.data());
        }

        if (idraw < 4)
        {
            Draw2(idraw);
        }
    }

private:
    void Draw2(int32_t idraw)
    {
        for (int32_t x = 0; x < m_sideLen; x++)
        {
            for (int32_t y = 0; y < m_sideLen; y++)
            {
                double r = m_cells[TUMOR]->Get(x, y) * 1.0 / m_latticeCap;
                double g = r;
                double b = r;
                r += m_cells[M1]->Get(x, y) * 1.0 / m_latticeCap;
                g += m_cells[M2]->Get(x, y) * 1.0 / m_latticeCap;
                b += m_cells[MBOTH]->Get(x, y) * 1.0 / m_latticeCap;
                m_win.SetPix(x + (m_sideLen + 1) * idraw, y, Hal::Rgb(r, g, b));
            }
        }
    }

    int32_t m_sideLen;
    double  m_divRate;
    double  m_divRateMut;
    double  m_divRateDoubleMut;
    double  m_deathRate;
    double  m_migrationRate;
    int32_t m_latticeCap;
    double  m_mutProb1;
    double  m_mutProb2;

    Hal::Rand                                                    m_rng;
    Hal::MultinomialCalc                                         m_cellMn;
    std::vector<std::unique_ptr<Hal::PopulationGrid2D<int32_t>>> m_cells;
    std::vector<int32_t>                                         m_sums;
    Hal::Examples::GridWindow                                   &m_win;
};

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t sideLen = 150;
        const int32_t drawScale = 2;
        Hal::Examples::GridWindow win("Population Tumor", (sideLen + 1) * 4, sideLen, drawScale, !headless);
        win.Clear(Hal::WHITE);

        PopTumor ft(sideLen, 0.010, 0.001, 0.008, 100000, 2e-6, 2e-6, win);
        ft.Init(1000);

        int32_t idraw = -1;
        for (int32_t i = 0; (i < 4001) && !win.IsClosed(); i++)
        {
            if ((i % 1000) == 0)
            {
                idraw++;
                std::printf("%lld\n", static_cast<long long>(ft.TotalPop()));
            }
            ft.Step(idraw);
            win.TickPause(0);
        }

        if (headless)
        {
            win.ToPng("poptumor.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("PopTumor failed: ") + e.what());
        return 1;
    }
}
