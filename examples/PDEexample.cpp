// A block of source agents and scattered sink agents impose fixed
// concentrations on a diffusible PDE field, stirred by time-varying
// circular advection and a pulsing diffusion rate. Sources and sinks are
// drawn on the left, the concentration field on the right.
//
// Live window by default; pass "headless" to dump the final frame to
// pdeexample.png.

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Grids/PDEGrid2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Rand.h"

#include "common/GridWindow.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <numeric>
#include <string>
#include <vector>

class PDEexample;

class SrcOrSink : public Hal::AgentSQ2DUnstackable<PDEexample>
{
public:
    explicit SrcOrSink(PDEexample *pGrid)
        : Hal::AgentSQ2DUnstackable<PDEexample>(pGrid),
          m_type(0)
    {
    }

    void    Init(int32_t t) { m_type = t; }
    int32_t Type() const { return m_type; }
    void Reaction();

private:
    int32_t m_type;
};

class PDEexample : public Hal::AgentGrid2D<SrcOrSink, PDEexample>
{
public:
    static const int32_t SRC;
    static const int32_t SINK;

    PDEexample(int32_t x,
               int32_t y)
        : Hal::AgentGrid2D<SrcOrSink, PDEexample>(x, y, true, true),
          m_diff(x, y)
    {
    }

    void Setup(int32_t nSinks,
               int32_t sinkDist)
    {
        for (int32_t x = Xdim() / 2; x < (Xdim() / 2 + 3); x++)
        {
            for (int32_t y = Ydim() / 2; y < (Ydim() / 2 + 3); y++)
            {
                NewAgentSQ(x, y).Init(SRC);
            }
        }
        std::vector<int32_t> sinkIs(Length());
        std::iota(sinkIs.begin(), sinkIs.end(), 0);
        m_rng.Shuffle(sinkIs.data(), static_cast<int32_t>(sinkIs.size()));

        int32_t sinksPlaced = 0;
        for (int32_t sinkI : sinkIs)
        {
            if (DistSquared(ItoX(sinkI), ItoY(sinkI), Xdim() / 2, Ydim() / 2) > (sinkDist * sinkDist))
            {
                NewAgentSQ(sinkI).Init(SINK);
                sinksPlaced++;
                if (sinksPlaced == nSinks)
                {
                    return;
                }
            }
        }
    }

    void Step(int32_t stepI)
    {
        double advectionX = std::sin(stepI * 1.0 / 1000) * 0.2;
        double advectionY = std::cos(stepI * 1.0 / 1000) * 0.2;
        for (SrcOrSink &srcOrSink : *this)
        {
            srcOrSink.Reaction();
        }
        m_diff.Advection(advectionX, advectionY, 0.0);
        double drate = (std::sin(stepI * 1.0 / 250) + 1) * 0.05;
        m_diff.Diffusion(drate);
        m_diff.Update();
    }

    void Draw(Hal::Examples::GridWindow &vis, int32_t diffColOffset)
    {
        for (SrcOrSink &srcOrSink : *this)
        {
            vis.SetPix(srcOrSink.Isq(), srcOrSink.Type());
        }
        for (int32_t x = 0; x < Xdim(); x++)
        {
            for (int32_t y = 0; y < Ydim(); y++)
            {
                vis.SetPix(diffColOffset + x, y, Hal::HeatMapRgb(m_diff.Get(x, y) * 4));
            }
        }
    }

    Hal::PDEGrid2D &Diff() { return m_diff; }

private:
    Hal::PDEGrid2D m_diff;
    Hal::Rand      m_rng;
};

const int32_t PDEexample::SRC = Hal::Rgb(0.0, 1.0, 0.0);
const int32_t PDEexample::SINK = Hal::Rgb(0.0, 1.0, 1.0);

void SrcOrSink::Reaction()
{
    this->Grid()->Diff().Set(Isq(), m_type == PDEexample::SRC ? 1.0 : 0.0);
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t x = 200;
        const int32_t y = 200;
        const int32_t scale = 2;

        Hal::Examples::GridWindow vis("PDE source/sink + advection", x * 2 + 1, y, scale, !headless);
        PDEexample ex(x, y);
        ex.Setup(100, 10);

        int32_t steps = headless ? 400 : 1000000;
        for (int32_t i = 0; (i < steps) && !vis.IsClosed(); i++)
        {
            ex.Step(i);
            ex.Draw(vis, x + 1);
            vis.TickPause(headless ? 0 : 30);
        }

        if (headless)
        {
            vis.ToPng("pdeexample.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("PDEexample failed: ") + e.what());
        return 1;
    }
}
