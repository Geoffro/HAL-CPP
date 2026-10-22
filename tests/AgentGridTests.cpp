#include <doctest/doctest.h>

#include "TestSupport.h"

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/IO/BinaryReader.h"
#include "Hal/IO/BinaryWriter.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal agent for exercising the pool directly.

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

    PoolTestGrid(int32_t xDim,
                 int32_t yDim,
                 bool    wrapX,
                 bool    wrapY)
        : Hal::AgentGrid2D<PoolTestAgent, PoolTestGrid>(xDim, yDim, wrapX, wrapY)
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
        if ((agent.Xsq() == 0) && (agent.Ysq() == 0))
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

// Game of Life as an end-to-end check of MapHood, GetAgent, and neighbor
// counting against a blinker, whose period-2 oscillation is known.
//
// GOLAgent and GOLGrid reference each other, so both are forward-declared,
// GOLAgent is defined first with out-of-line methods, and GOLGrid second.

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
        // The tick advances before stepping, so GetCurrState() reads the
        // slot the previous call wrote as "next".
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
    const std::string path = TestHelpers::TempFilePath("hal_test_agentpool_roundtrip.bin");

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

TEST_CASE("AgentSQ2DUnstackable::MoveSQ relocates an agent and frees its old square")
{
    PoolTestGrid grid(5, 5);
    PoolTestAgent &agent = grid.NewAgentSQ(1, 1);

    agent.MoveSQ(3, 2);
    CHECK(agent.Xsq() == 3);
    CHECK(agent.Ysq() == 2);
    CHECK(agent.Isq() == grid.I(3, 2));
    CHECK(grid.GetAgent(3, 2) == &agent);
    CHECK(grid.GetAgent(1, 1) == nullptr);

    agent.MoveSQ(grid.I(0, 4));
    CHECK(agent.Xsq() == 0);
    CHECK(agent.Ysq() == 4);
    CHECK(grid.GetAgent(0, 4) == &agent);
    CHECK(grid.GetAgent(3, 2) == nullptr);

    grid.NewAgentSQ(2, 2);
    CHECK_THROWS_AS(agent.MoveSQ(2, 2), std::runtime_error);
}

TEST_CASE("AgentSQ2DUnstackable::MoveSafeSQ wraps on wrapped axes and stays put on bounded ones")
{
    PoolTestGrid grid(5, 5, true, false);
    PoolTestAgent &agent = grid.NewAgentSQ(0, 0);

    agent.MoveSafeSQ(-1, -1);
    CHECK(agent.Xsq() == 4);
    CHECK(agent.Ysq() == 0);

    agent.MoveSafeSQ(5, 7);
    CHECK(agent.Xsq() == 0);
    CHECK(agent.Ysq() == 0);
    CHECK(grid.GetAgent(0, 0) == &agent);
    CHECK(grid.Pop() == 1);
}

TEST_CASE("AgentSQ2DUnstackable::SwapPosition exchanges two agents' squares")
{
    PoolTestGrid grid(4, 4);
    PoolTestAgent &a = grid.NewAgentSQ(0, 1);
    PoolTestAgent &b = grid.NewAgentSQ(3, 2);

    a.SwapPosition(b);
    CHECK(a.Xsq() == 3);
    CHECK(a.Ysq() == 2);
    CHECK(b.Xsq() == 0);
    CHECK(b.Ysq() == 1);
    CHECK(grid.GetAgent(3, 2) == &a);
    CHECK(grid.GetAgent(0, 1) == &b);
}

TEST_CASE("AgentSQ2DUnstackable: disposing an already dead agent throws and leaves the grid intact")
{
    PoolTestGrid grid(3, 3);
    PoolTestAgent &agent = grid.NewAgentSQ(1, 1);
    agent.Dispose();

    PoolTestAgent &replacement = grid.NewAgentSQ(1, 1);
    CHECK(&replacement == &agent);
    replacement.Dispose();
    CHECK_THROWS_AS(agent.Dispose(), std::runtime_error);
    CHECK(grid.Pop() == 0);

    PoolTestAgent &other = grid.NewAgentSQ(2, 2);
    CHECK(grid.Pop() == 1);
    CHECK(grid.GetAgent(2, 2) == &other);
}

TEST_CASE("AgentGrid2D::ShuffleAgents permutes the iteration order and keeps every agent")
{
    PoolTestGrid grid(10, 10);
    for (int32_t i = 0; i < 50; i++)
    {
        grid.NewAgentSQ(i).SetTag(i);
    }

    Hal::Rand rand(3);
    grid.ShuffleAgents(rand);

    std::vector<int32_t> tags;
    for (PoolTestAgent &agent : grid)
    {
        tags.push_back(agent.Tag());
    }
    REQUIRE(tags.size() == 50);
    std::vector<int32_t> sorted = tags;
    std::sort(sorted.begin(), sorted.end());
    for (int32_t i = 0; i < 50; i++)
    {
        CHECK(sorted[static_cast<size_t>(i)] == i);
    }
    CHECK(!std::is_sorted(tags.begin(), tags.end()));
}

TEST_CASE("AgentGrid2D::GetAgentSafe and GetAgentsSafe wrap or reject out-of-bounds coordinates")
{
    PoolTestGrid wrapped(4, 4, true, true);
    PoolTestAgent &agent = wrapped.NewAgentSQ(3, 0);
    CHECK(wrapped.GetAgentSafe(-1, 4) == &agent);
    std::vector<PoolTestAgent *> found;
    wrapped.GetAgentsSafe(found, -1, 0);
    REQUIRE(found.size() == 1);
    CHECK(found[0] == &agent);

    PoolTestGrid bounded(4, 4);
    bounded.NewAgentSQ(0, 0);
    CHECK(bounded.GetAgentSafe(-1, 0) == nullptr);
    found.clear();
    bounded.GetAgentsSafe(found, 0, 4);
    CHECK(found.empty());
}

TEST_CASE("AgentBase::Age counts ticks since birth")
{
    PoolTestGrid grid(3, 3);
    grid.IncTick();
    PoolTestAgent &agent = grid.NewAgentSQ(0, 0);
    CHECK(agent.BirthTick() == 1);
    grid.IncTick();
    grid.IncTick();
    CHECK(agent.Age() == 2);
}

TEST_CASE("AgentGrid2D::SaveAgents / LoadAgents restore the tick and each agent's birth tick")
{
    const std::string path = TestHelpers::TempFilePath("hal_test_agentpool_ticks.bin");

    PoolTestGrid original(4, 4);
    original.NewAgentSQ(0, 0);
    original.IncTick();
    original.IncTick();
    original.NewAgentSQ(1, 1);
    original.IncTick();

    {
        Hal::BinaryWriter writer(path);
        original.SaveAgents(writer, [](PoolTestAgent &, Hal::BinaryWriter &) {});
    }

    PoolTestGrid loaded(4, 4);
    {
        Hal::BinaryReader reader(path);
        loaded.LoadAgents(reader, [](PoolTestAgent &, Hal::BinaryReader &) {});
        CHECK(reader.IsGood());
    }

    CHECK(loaded.GetTick() == 3);
    REQUIRE(loaded.GetAgent(0, 0) != nullptr);
    REQUIRE(loaded.GetAgent(1, 1) != nullptr);
    CHECK(loaded.GetAgent(0, 0)->Age() == 3);
    CHECK(loaded.GetAgent(1, 1)->Age() == 1);

    std::remove(path.c_str());
}

static void LoadNoFields(PoolTestAgent     &agent,
                         Hal::BinaryReader &reader)
{
    (void)agent;
    (void)reader;
}

static void WriteAgentHeader(const std::string &path,
                             int32_t            pop,
                             int32_t            iSq)
{
    Hal::BinaryWriter writer(path);
    writer.WriteInt32(0);
    writer.WriteInt32(pop);
    writer.WriteInt32(iSq);
    writer.WriteInt32(0);
}

TEST_CASE("AgentGrid2D::LoadAgents rejects truncated data, negative populations, and out-of-range squares")
{
    const std::string path = TestHelpers::TempFilePath("hal_test_agentpool_invalid.bin");

    // Claims two agents but holds one.
    WriteAgentHeader(path, 2, 5);
    {
        PoolTestGrid grid(4, 4);
        Hal::BinaryReader reader(path);
        CHECK_THROWS_AS(grid.LoadAgents(reader, LoadNoFields), std::runtime_error);
    }

    WriteAgentHeader(path, -1, 5);
    {
        PoolTestGrid grid(4, 4);
        Hal::BinaryReader reader(path);
        CHECK_THROWS_AS(grid.LoadAgents(reader, LoadNoFields), std::runtime_error);
        CHECK(grid.Pop() == 0);
    }

    WriteAgentHeader(path, 1, 16);
    {
        PoolTestGrid grid(4, 4);
        Hal::BinaryReader reader(path);
        CHECK_THROWS_AS(grid.LoadAgents(reader, LoadNoFields), std::runtime_error);
        CHECK(grid.Pop() == 0);
    }

    std::remove(path.c_str());
}
