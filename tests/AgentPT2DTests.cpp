#include <doctest/doctest.h>

#include "Hal/Agents/AgentPT2D.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Rand.h"

#include <algorithm>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------
// Off-lattice, stackable agents, ported from Examples-style usage of
// AgentPT2D. See AgentGridTests.cpp for why GOLAgent/GOLGrid-style mutual
// references need the forward-declare-both / define-agent-first pattern —
// the same requirement applies here.
// ---------------------------------------------------------------------

class ParticleGrid;

class ParticleAgent : public Hal::AgentPT2D<ParticleGrid>
{
public:
    explicit ParticleAgent(ParticleGrid *pGrid)
        : Hal::AgentPT2D<ParticleGrid>(pGrid),
          m_tag(0)
    {
    }

    ~ParticleAgent()
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

class ParticleGrid : public Hal::AgentGrid2D<ParticleAgent, ParticleGrid>
{
public:
    ParticleGrid(int32_t xDim,
                 int32_t yDim)
        : Hal::AgentGrid2D<ParticleAgent, ParticleGrid>(xDim, yDim)
    {
    }

    ParticleGrid(int32_t xDim,
                 int32_t yDim,
                 bool    wrapX,
                 bool    wrapY)
        : Hal::AgentGrid2D<ParticleAgent, ParticleGrid>(xDim, yDim, wrapX, wrapY)
    {
    }

    ~ParticleGrid()
    {
    }
};

TEST_CASE("AgentPT2D: multiple agents can stack on the same square")
{
    ParticleGrid grid(10, 10);
    ParticleAgent &a = grid.NewAgentPT(3.5, 3.5);
    ParticleAgent &b = grid.NewAgentPT(3.2, 3.9);
    a.SetTag(1);
    b.SetTag(2);

    CHECK(a.Xsq() == 3);
    CHECK(a.Ysq() == 3);
    CHECK(b.Xsq() == 3);
    CHECK(b.Ysq() == 3);
    CHECK(a.GetCountOnSquare() == 2);
    CHECK(b.GetCountOnSquare() == 2);

    std::vector<ParticleAgent *> onSquare;
    grid.GetAgents(onSquare, 3, 3);
    REQUIRE(onSquare.size() == 2);
    // Most-recently-added occupant is threaded at the head of the list.
    CHECK(onSquare[0] == &b);
    CHECK(onSquare[1] == &a);
}

TEST_CASE("AgentPT2D: disposing one stacked agent relinks the remaining occupant correctly")
{
    ParticleGrid grid(10, 10);
    ParticleAgent &a = grid.NewAgentPT(3.5, 3.5);
    ParticleAgent &b = grid.NewAgentPT(3.2, 3.9);
    a.SetTag(1);
    b.SetTag(2);

    b.Dispose();

    CHECK(a.GetCountOnSquare() == 1);
    std::vector<ParticleAgent *> onSquare;
    grid.GetAgents(onSquare, 3, 3);
    REQUIRE(onSquare.size() == 1);
    CHECK(onSquare[0] == &a);
    CHECK(grid.GetAgent(3, 3) == &a);
}

TEST_CASE("AgentPT2D: MovePT relocates an agent to a new square and updates counts")
{
    ParticleGrid grid(10, 10);
    ParticleAgent &a = grid.NewAgentPT(1.5, 1.5);
    CHECK(a.GetCountOnSquare() == 1);

    a.MovePT(7.2, 8.9);
    CHECK(a.Xsq() == 7);
    CHECK(a.Ysq() == 8);
    CHECK(grid.GetAgent(1, 1) == nullptr);
    CHECK(grid.GetAgent(7, 8) == &a);
}

TEST_CASE("AgentGrid2D::GetAgentsRad matches a brute-force distance check over all agents")
{
    ParticleGrid grid(20, 20, true, true);
    Hal::Rand rand(2024);

    const int32_t agentCount = 200;
    std::vector<ParticleAgent *> allAgents;
    for (int32_t i = 0; i < agentCount; i++)
    {
        double x = rand.Double(20.0);
        double y = rand.Double(20.0);
        ParticleAgent &a = grid.NewAgentPT(x, y);
        a.SetTag(i);
        allAgents.push_back(&a);
    }

    // A handful of query points/radii, including ones near the wrapped
    // boundary so wraparound distance handling gets exercised too.
    struct Query
    {
        double x;
        double y;
        double rad;
    };
    std::vector<Query> queries = {
        {10.0, 10.0, 3.0},
        {0.2, 0.3, 4.0},
        {19.8, 19.7, 5.0},
        {5.0, 15.0, 6.5}};

    for (const Query &q : queries)
    {
        std::vector<ParticleAgent *> found;
        grid.GetAgentsRad(found, q.x, q.y, q.rad);

        std::vector<int32_t> foundTags;
        for (ParticleAgent *pAgent : found)
        {
            foundTags.push_back(pAgent->Tag());
        }
        std::sort(foundTags.begin(), foundTags.end());

        std::vector<int32_t> expectedTags;
        for (ParticleAgent *pAgent : allAgents)
        {
            double dx = grid.DispX(q.x, pAgent->Xpt());
            double dy = grid.DispY(q.y, pAgent->Ypt());
            if (std::sqrt(dx * dx + dy * dy) < q.rad)
            {
                expectedTags.push_back(pAgent->Tag());
            }
        }
        std::sort(expectedTags.begin(), expectedTags.end());

        CAPTURE(q.x);
        CAPTURE(q.y);
        CAPTURE(q.rad);
        CHECK(foundTags == expectedTags);
        CHECK_FALSE(expectedTags.empty());
    }
}

TEST_CASE("AgentGrid2D::GetAgentsRad EvalAgent overloads keep only predicate-accepted agents")
{
    ParticleGrid grid(20, 20, true, true);
    Hal::Rand rand(77);

    const int32_t agentCount = 150;
    std::vector<ParticleAgent *> allAgents;
    for (int32_t i = 0; i < agentCount; i++)
    {
        ParticleAgent &a = grid.NewAgentPT(rand.Double(20.0), rand.Double(20.0));
        a.SetTag(i);
        allAgents.push_back(&a);
    }

    const double queryX = 4.5;
    const double queryY = 16.0;
    const double rad = 5.0;
    auto evenTag = [](ParticleAgent *pAgent)
    {
        return (pAgent->Tag() % 2) == 0;
    };

    std::vector<ParticleAgent *> found;
    grid.GetAgentsRad(found, queryX, queryY, rad, evenTag);

    std::vector<int32_t> foundTags;
    for (ParticleAgent *pAgent : found)
    {
        foundTags.push_back(pAgent->Tag());
    }
    std::sort(foundTags.begin(), foundTags.end());

    std::vector<int32_t> expectedTags;
    for (ParticleAgent *pAgent : allAgents)
    {
        double dx = grid.DispX(queryX, pAgent->Xpt());
        double dy = grid.DispY(queryY, pAgent->Ypt());
        if ((std::sqrt(dx * dx + dy * dy) < rad) && ((pAgent->Tag() % 2) == 0))
        {
            expectedTags.push_back(pAgent->Tag());
        }
    }
    std::sort(expectedTags.begin(), expectedTags.end());

    CHECK(foundTags == expectedTags);
    CHECK_FALSE(expectedTags.empty());

    // The filtered approximate scan is a superset of the filtered exact
    // search and contains no predicate-rejected agents.
    std::vector<ParticleAgent *> approx;
    grid.GetAgentsRadApprox(approx, queryX, queryY, rad, evenTag);
    CHECK(approx.size() >= found.size());
    for (ParticleAgent *pAgent : approx)
    {
        CHECK((pAgent->Tag() % 2) == 0);
    }
}

TEST_CASE("AgentGrid2D::GetAgentsRad displacement-info overloads record wrap-aware distance and displacement per found agent")
{
    ParticleGrid grid(20, 20, true, true);
    Hal::Rand rand(99);

    const int32_t agentCount = 120;
    for (int32_t i = 0; i < agentCount; i++)
    {
        ParticleAgent &a = grid.NewAgentPT(rand.Double(20.0), rand.Double(20.0));
        a.SetTag(i);
    }

    // Query near the wrapped corner so displacements must take the short
    // way around.
    const double queryX = 0.5;
    const double queryY = 19.5;
    const double rad = 4.0;

    std::vector<ParticleAgent *> found;
    std::vector<Hal::AgentDisplacementInfo2D> info;
    grid.GetAgentsRad(found, info, queryX, queryY, rad);

    REQUIRE(info.size() == found.size());
    CHECK_FALSE(found.empty());
    for (size_t k = 0; k < found.size(); k++)
    {
        ParticleAgent *pAgent = found[k];
        double dx = grid.DispX(queryX, pAgent->Xpt());
        double dy = grid.DispY(queryY, pAgent->Ypt());
        CHECK(info[k].dispX == doctest::Approx(dx));
        CHECK(info[k].dispY == doctest::Approx(dy));
        CHECK(info[k].dist == doctest::Approx(std::sqrt(dx * dx + dy * dy)));
        CHECK(info[k].dist < rad);
    }

    // Padding behavior when the output lists already hold entries from an
    // earlier query (Java aligns displacementInfo[i] with retAgentList[i]
    // by padding): re-run the same query into the same vectors.
    size_t firstCount = found.size();
    grid.GetAgentsRad(found, info, queryX, queryY, rad);
    REQUIRE(found.size() == 2 * firstCount);
    REQUIRE(info.size() == found.size());
    for (size_t k = firstCount; k < found.size(); k++)
    {
        CHECK(found[k] == found[k - firstCount]);
        CHECK(info[k].dist == doctest::Approx(info[k - firstCount].dist));
    }

    // Filtered displacement variant agrees with filtering the unfiltered
    // result.
    std::vector<ParticleAgent *> filteredFound;
    std::vector<Hal::AgentDisplacementInfo2D> filteredInfo;
    auto evenTag = [](ParticleAgent *pAgent)
    {
        return (pAgent->Tag() % 2) == 0;
    };
    grid.GetAgentsRad(filteredFound, filteredInfo, queryX, queryY, rad, evenTag);
    size_t evenCount = 0;
    for (size_t k = 0; k < firstCount; k++)
    {
        if ((found[k]->Tag() % 2) == 0)
        {
            evenCount++;
        }
    }
    CHECK(filteredFound.size() == evenCount);
    for (ParticleAgent *pAgent : filteredFound)
    {
        CHECK((pAgent->Tag() % 2) == 0);
    }
}
