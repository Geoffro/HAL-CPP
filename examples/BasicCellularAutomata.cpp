// Ported from Examples/BasicCellularAutomata: a one-dimensional elementary
// cellular automaton (Wolfram rule, default 35) grown row by row down the
// grid, each new row's cells kept or removed by the rule applied to the
// three cells above. The result is the classic self-similar triangle.
//
// The Java version records a GIF; here the live window animates the growth
// and the headless run dumps the final triangle to basiccellularautomata.png.

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

class BasicCellularAutomata;

class BasicCell : public Hal::AgentSQ2DUnstackable<BasicCellularAutomata>
{
public:
    explicit BasicCell(BasicCellularAutomata *pGrid)
        : Hal::AgentSQ2DUnstackable<BasicCellularAutomata>(pGrid)
    {
    }

    int32_t DetermineNeighbors();
};

class BasicCellularAutomata : public Hal::AgentGrid2D<BasicCell, BasicCellularAutomata>
{
public:
    static constexpr int32_t RULE = 35;        // 0..255
    static constexpr int32_t DOMAIN_SIZE = 200;

    explicit BasicCellularAutomata(Hal::Examples::GridWindow &vis)
        : Hal::AgentGrid2D<BasicCell, BasicCellularAutomata>(DOMAIN_SIZE * 2 - 1, DOMAIN_SIZE, false, false),
          m_vis(vis)
    {
        NewAgentSQ((Xdim() - 1) / 2, Ydim() - 1);
    }

    void StepCells(int32_t row)
    {
        int32_t middleX = (Xdim() - 1) / 2;
        for (int32_t x = middleX - (row - 1); x <= (middleX + (row - 1)); x++)
        {
            BasicCell &c = NewAgentSQ(x, Ydim() - row);
            if (GetBit(RULE, c.DetermineNeighbors()) == 0)
            {
                c.Dispose();
            }
        }
    }

    void Draw()
    {
        for (int32_t i = 0; i < Length(); i++)
        {
            BasicCell *pCell = GetAgent(i);
            m_vis.SetPix(i, (pCell != nullptr) ? BLACK : WHITE);
        }
    }

    static int32_t GetBit(int32_t integer, int32_t i)
    {
        return (integer >> (7 - i)) & 1;
    }

    static const int32_t BLACK;
    static const int32_t WHITE;

private:
    Hal::Examples::GridWindow &m_vis;
};

const int32_t BasicCellularAutomata::BLACK = Hal::BLACK;
const int32_t BasicCellularAutomata::WHITE = Hal::WHITE;

int32_t BasicCell::DetermineNeighbors()
{
    BasicCellularAutomata *pGrid = this->Grid();
    BasicCell *aboveLeft = (Xsq() - 1 >= 0) ? pGrid->GetAgent(Xsq() - 1, Ysq() + 1) : nullptr;
    BasicCell *aboveCenter = pGrid->GetAgent(Xsq(), Ysq() + 1);
    BasicCell *aboveRight = (Xsq() + 1 <= pGrid->Xdim() - 1) ? pGrid->GetAgent(Xsq() + 1, Ysq() + 1) : nullptr;

    bool l = (aboveLeft != nullptr);
    bool c = (aboveCenter != nullptr);
    bool r = (aboveRight != nullptr);
    // Encode the (left,center,right) occupancy triple as 0..7, MSB = left.
    return (l ? 0 : 4) + (c ? 0 : 2) + (r ? 0 : 1);
}

int main(int argc, char **argv)
{
    try
    {
        bool headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);

        const int32_t xDim = BasicCellularAutomata::DOMAIN_SIZE * 2 - 1;
        const int32_t yDim = BasicCellularAutomata::DOMAIN_SIZE;
        int32_t       scale = (xDim > 300) ? 1 : 3;

        Hal::Examples::GridWindow vis("Elementary CA (rule 35)", xDim, yDim, scale, !headless);
        BasicCellularAutomata grid(vis);
        grid.Draw();

        for (int32_t tick = 2; (tick <= yDim) && !vis.IsClosed(); tick++)
        {
            vis.TickPause(0);
            grid.StepCells(tick);
            grid.Draw();
        }

        if (headless)
        {
            vis.ToPng("basiccellularautomata.png");
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("BasicCellularAutomata failed: ") + e.what());
        return 1;
    }
}
