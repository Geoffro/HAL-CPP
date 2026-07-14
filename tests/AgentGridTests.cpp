#include <doctest/doctest.h>

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/IO/BinaryReader.h"
#include "Hal/IO/BinaryWriter.h"
#include "Hal/Neighborhood.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------
// Minimal agent used to exercise the object pool directly (birth, death,
// iteration skipping dead/newly-born agents, CleanAgents compaction) —
// independent of any model-specific logic.
// ---------------------------------------------------------------------

class PoolTestGrid;

class PoolTestAgent : public Hal::AgentSQ2DUnstackable<PoolTestGrid>
{
public:
    explicit PoolTestAgent(PoolTestGrid *pGrid)
        : Hal::AgentSQ2DUnstackable<PoolTestGrid>(pGrid),
          m_tag(0)
    {
    }

    ~PoolTestAgent()
    {
    }

    void SetTag(int32_t tag)
    {
        m_tag = tag;
    }

    int32_t Tag() const
    {
        return m_tag;
    }

private:
    int32_t m_tag;
};

class PoolTestGrid : public Hal::AgentGrid2D<PoolTestAgent, PoolTestGrid>
{
public:
    PoolTestGrid(int32_t xDim,
                 int32_t yDim)
        : Hal::AgentGrid2D<PoolTestAgent, PoolTestGrid>(xDim, yDim)
    {
    }

    ~PoolTestGrid()
    {
    }
};

TEST_CASE("AgentGrid2D: NewAgentSQ places one agent per square and tracks population")
{
    PoolTestGrid grid(3, 3);
    CHECK(grid.Pop() == 0);

    PoolTestAgent &a = grid.NewAgentSQ(1, 1);
    a.SetTag(42);
    CHECK(grid.Pop() == 1);
    CHECK(grid.GetAgent(1, 1) == &a);
    CHECK(grid.GetAgent(1, 1)->Tag() == 42);
    CHECK(grid.GetAgent(0, 0) == nullptr);
}

TEST_CASE("AgentGrid2D: adding a second agent to an occupied square throws")
{
    PoolTestGrid grid(3, 3);
    grid.NewAgentSQ(1, 1);
    CHECK_THROWS_AS(grid.NewAgentSQ(1, 1), std::runtime_error);
}

TEST_CASE("AgentGrid2D: Dispose frees the square and iteration skips dead agents")
{
    PoolTestGrid grid(3, 3);
    PoolTestAgent &a = grid.NewAgentSQ(0, 0);
    grid.NewAgentSQ(1, 1);
    CHECK(grid.Pop() == 2);

    a.Dispose();
    CHECK(grid.Pop() == 1);
    CHECK(grid.GetAgent(0, 0) == nullptr);

    int32_t visited = 0;
    for (PoolTestAgent &agent : grid)
    {
        visited++;
        CHECK(agent.IsAlive());
    }
    CHECK(visited == 1);
}

TEST_CASE("AgentGrid2D: a disposed slot is recycled by the next NewAgentSQ")
{
    PoolTestGrid grid(3, 3);
    PoolTestAgent &a = grid.NewAgentSQ(0, 0);
    a.Dispose();

    PoolTestAgent &b = grid.NewAgentSQ(2, 2);
    CHECK(&b == &a);
    CHECK(grid.GetAgent(2, 2) == &b);
}

TEST_CASE("AgentGrid2D: agents born during iteration are not visited in the same pass")
{
    PoolTestGrid grid(3, 3);
    grid.NewAgentSQ(0, 0);
    grid.NewAgentSQ(0, 1);

    int32_t visited = 0;
    for (PoolTestAgent &agent : grid)
    {
        visited++;
        if (agent.Xsq() == 0 && agent.Ysq() == 0)
        {
            grid.NewAgentSQ(2, 2);
        }
    }
    CHECK(visited == 2);
    CHECK(grid.Pop() == 3);
}

TEST_CASE("AgentGrid2D: CleanAgents compacts dead slots and a subsequent iteration sees only alive agents")
{
    PoolTestGrid grid(4, 4);
    PoolTestAgent &a = grid.NewAgentSQ(0, 0);
    grid.NewAgentSQ(1, 1);
    grid.NewAgentSQ(2, 2);
    a.Dispose();
    grid.CleanAgents();

    CHECK(grid.Pop() == 2);
    int32_t visited = 0;
    for (PoolTestAgent &agent : grid)
    {
        visited++;
        CHECK(agent.IsAlive());
    }
    CHECK(visited == 2);
}

// ---------------------------------------------------------------------
// Game of Life, ported from Examples/GameOfLife/GOLGrid.java, as an
// end-to-end check of MapHood + GetAgent + neighbor counting against a
// known oscillator (a blinker) rather than against captured Java output —
// the blinker's period-2 oscillation is the definition of Conway's Game
// of Life, independent of this port entirely.
//
// NOTE: GOLAgent and GOLGrid mutually reference each other (the agent
// calls back into the grid for MapHood/GetAgent/GetTick, and the grid's
// AgentGrid2D<GOLAgent> base stores GOLAgent objects by value). Unlike
// Java, where declaration order never matters, C++ requires the forward-
// declare-both / define-GOLAgent-first-with-out-of-line-methods /
// define-GOLGrid-second pattern below whenever two classes reference each
// other this way — worth keeping in mind for every future on-lattice
// model ported into this codebase.
// ---------------------------------------------------------------------

class GOLGrid;

class GOLAgent : public Hal::AgentSQ2DUnstackable<GOLGrid>
{
public:
    explicit GOLAgent(GOLGrid *pGrid);
    ~GOLAgent();

    int32_t GetCurrState() const;
    void SetNextState(int32_t state);
    void Step();

private:
    int32_t m_state[2];
};

class GOLGrid : public Hal::AgentGrid2D<GOLAgent, GOLGrid>
{
public:
    GOLGrid(int32_t xDim,
            int32_t yDim)
        : Hal::AgentGrid2D<GOLAgent, GOLGrid>(xDim, yDim, false, false),
          m_mooreHood(Hal::MooreHood(false))
    {
        for (int32_t i = 0; i < Length(); i++)
        {
            NewAgentSQ(i);
        }
    }

    ~GOLGrid()
    {
    }

    int32_t *MooreHoodData()
    {
        return m_mooreHood.data();
    }

    int32_t MooreHoodSize() const
    {
        return static_cast<int32_t>(m_mooreHood.size());
    }

    void StepAgents()
    {
        // Tick advances before stepping, matching Java's StepAgents (tick++
        // first): GetCurrState() reads the buffer slot written as "next" by
        // the previous call (or by initial setup, for the first call), so
        // the state visible via GetCurrState() always lags one StepAgents()
        // call behind the neighbor computation that produced it.
        IncTick();
        for (GOLAgent &agent : *this)
        {
            agent.Step();
        }
    }

private:
    std::vector<int32_t> m_mooreHood;
};

GOLAgent::GOLAgent(GOLGrid *pGrid)
    : Hal::AgentSQ2DUnstackable<GOLGrid>(pGrid),
      m_state{0, 0}
{
}

GOLAgent::~GOLAgent()
{
}

int32_t GOLAgent::GetCurrState() const
{
    return m_state[this->Grid()->GetTick() % 2];
}

void GOLAgent::SetNextState(int32_t state)
{
    m_state[(this->Grid()->GetTick() + 1) % 2] = state;
}

void GOLAgent::Step()
{
    GOLGrid *pGrid = this->Grid();
    int32_t hoodLen = MapHood(pGrid->MooreHoodData(), pGrid->MooreHoodSize());
    int32_t countNeighbors = 0;
    for (int32_t i = 0; i < hoodLen; i++)
    {
        if (pGrid->GetAgent(pGrid->MooreHoodData()[i])->GetCurrState() == 1)
        {
            countNeighbors++;
        }
    }
    bool alive = (GetCurrState() == 1);
    if ((alive && ((countNeighbors == 2) || (countNeighbors == 3))) || (!alive && (countNeighbors == 3)))
    {
        SetNextState(1);
    }
    else
    {
        SetNextState(0);
    }
}

TEST_CASE("GameOfLife: a vertical blinker oscillates to horizontal and back, matching Conway's rules")
{
    GOLGrid grid(5, 5);

    // Vertical blinker centered at column 2, rows 1-3. Written into the
    // "next" buffer slot at construction time (tick=0); it only becomes
    // visible via GetCurrState() once the first StepAgents() call advances
    // the tick (see the double-buffering note on StepAgents above).
    grid.GetAgent(2, 1)->SetNextState(1);
    grid.GetAgent(2, 2)->SetNextState(1);
    grid.GetAgent(2, 3)->SetNextState(1);

    auto checkLive = [&](std::initializer_list<std::pair<int32_t, int32_t>> liveCells)
    {
        for (int32_t x = 0; x < 5; x++)
        {
            for (int32_t y = 0; y < 5; y++)
            {
                bool expectLive = false;
                for (const auto &cell : liveCells)
                {
                    if ((cell.first == x) && (cell.second == y))
                    {
                        expectLive = true;
                    }
                }
                CAPTURE(x);
                CAPTURE(y);
                CHECK(grid.GetAgent(x, y)->GetCurrState() == (expectLive ? 1 : 0));
            }
        }
    };

    grid.StepAgents();
    checkLive({{2, 1}, {2, 2}, {2, 3}});

    grid.StepAgents();
    checkLive({{1, 2}, {2, 2}, {3, 2}});

    grid.StepAgents();
    checkLive({{2, 1}, {2, 2}, {2, 3}});
}

TEST_CASE("AgentGrid2D::MapEmptyHood / MapOccupiedHood split the Moore neighborhood by occupancy")
{
    PoolTestGrid grid(3, 3);
    // Occupy the center and one neighbor; leave the rest empty.
    grid.NewAgentSQ(1, 1);
    grid.NewAgentSQ(0, 1);

    std::vector<int32_t> hood = Hal::MooreHood(false);

    int32_t emptyCt = grid.MapEmptyHood(hood.data(), static_cast<int32_t>(hood.size()), 1, 1);
    // The 8 Moore neighbors of the center: (0,1) is occupied, 7 are empty.
    CHECK(emptyCt == 7);
    for (int32_t k = 0; k < emptyCt; k++)
    {
        CHECK(grid.GetAgent(hood[k]) == nullptr);
    }

    std::vector<int32_t> hood2 = Hal::MooreHood(false);
    int32_t occCt = grid.MapOccupiedHood(hood2.data(), static_cast<int32_t>(hood2.size()), 1, 1);
    CHECK(occCt == 1);
    CHECK(grid.GetAgent(hood2[0]) == grid.GetAgent(0, 1));
}

TEST_CASE("AgentGrid2D::PopAt reports occupancy per square for unstackable agents")
{
    PoolTestGrid grid(3, 3);
    CHECK(grid.PopAt(1, 1) == 0);
    grid.NewAgentSQ(1, 1);
    CHECK(grid.PopAt(1, 1) == 1);
    CHECK(grid.PopAt(grid.I(1, 1)) == 1);
    CHECK(grid.PopAt(0, 0) == 0);
}

TEST_CASE("AgentGrid2D::SaveAgents / LoadAgents round-trip agent positions and state through a file")
{
    const std::string path = "agentpool_roundtrip.bin";

    // Populate a grid with tagged agents at known squares.
    PoolTestGrid original(5, 5);
    original.NewAgentSQ(0, 0).SetTag(10);
    original.NewAgentSQ(2, 3).SetTag(20);
    original.NewAgentSQ(4, 4).SetTag(30);

    {
        Hal::BinaryWriter writer(path);
        original.SaveAgents(writer, [](PoolTestAgent &a, Hal::BinaryWriter &w) { w.WriteInt32(a.Tag()); });
    }

    // Load into a fresh, empty grid of matching dimensions.
    PoolTestGrid loaded(5, 5);
    {
        Hal::BinaryReader reader(path);
        loaded.LoadAgents(reader, [](PoolTestAgent &a, Hal::BinaryReader &r) { a.SetTag(r.ReadInt32()); });
    }

    CHECK(loaded.Pop() == 3);
    CHECK(loaded.GetAgent(0, 0) != nullptr);
    CHECK(loaded.GetAgent(0, 0)->Tag() == 10);
    CHECK(loaded.GetAgent(2, 3) != nullptr);
    CHECK(loaded.GetAgent(2, 3)->Tag() == 20);
    CHECK(loaded.GetAgent(4, 4) != nullptr);
    CHECK(loaded.GetAgent(4, 4)->Tag() == 30);
    CHECK(loaded.GetAgent(1, 1) == nullptr);

    std::remove(path.c_str());
}
