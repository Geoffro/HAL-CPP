#include <doctest/doctest.h>

#include "Hal/Agents/AgentPT3D.h"
#include "Hal/Agents/AgentSQ3DUnstackable.h"
#include "Hal/Grids/AgentGrid3D.h"
#include "Hal/Rand.h"

#include <algorithm>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------
// 3D on-lattice pool basics, mirroring AgentGridTests.cpp's PoolTestAgent
// coverage but in 3D.
// ---------------------------------------------------------------------

class PoolTestGrid3D;

class PoolTestAgent3D : public Hal::AgentSQ3DUnstackable<PoolTestGrid3D>
{
public:
    explicit PoolTestAgent3D(PoolTestGrid3D *pGrid)
        : Hal::AgentSQ3DUnstackable<PoolTestGrid3D>(pGrid)
    {
    }

    ~PoolTestAgent3D()
    {
    }
};

class PoolTestGrid3D : public Hal::AgentGrid3D<PoolTestAgent3D, PoolTestGrid3D>
{
public:
    PoolTestGrid3D(int32_t xDim,
                   int32_t yDim,
                   int32_t zDim)
        : Hal::AgentGrid3D<PoolTestAgent3D, PoolTestGrid3D>(xDim, yDim, zDim)
    {
    }

    ~PoolTestGrid3D()
    {
    }
};

TEST_CASE("AgentGrid3D: NewAgentSQ places one agent per voxel and Dispose frees it")
{
    PoolTestGrid3D grid(3, 3, 3);
    PoolTestAgent3D &a = grid.NewAgentSQ(1, 1, 1);
    CHECK(grid.Pop() == 1);
    CHECK(grid.GetAgent(1, 1, 1) == &a);
    CHECK(grid.GetAgent(0, 0, 0) == nullptr);

    a.Dispose();
    CHECK(grid.Pop() == 0);
    CHECK(grid.GetAgent(1, 1, 1) == nullptr);
}

TEST_CASE("AgentGrid3D: adding a second agent to an occupied voxel throws")
{
    PoolTestGrid3D grid(3, 3, 3);
    grid.NewAgentSQ(1, 1, 1);
    CHECK_THROWS_AS(grid.NewAgentSQ(1, 1, 1), std::runtime_error);
}

TEST_CASE("AgentGrid3D: iteration skips dead and mid-iteration-born agents")
{
    PoolTestGrid3D grid(3, 3, 3);
    grid.NewAgentSQ(0, 0, 0);
    PoolTestAgent3D &b = grid.NewAgentSQ(1, 1, 1);
    b.Dispose();

    int32_t visited = 0;
    for (PoolTestAgent3D &agent : grid)
    {
        visited++;
        CHECK(agent.IsAlive());
        grid.NewAgentSQ(2, 2, 2);
    }
    CHECK(visited == 1);
    CHECK(grid.Pop() == 2);
}

// ---------------------------------------------------------------------
// Off-lattice, stackable 3D agents + cell-list radius search, mirroring
// AgentPT2DTests.cpp's ParticleGrid coverage but in 3D.
// ---------------------------------------------------------------------

class ParticleGrid3D;

class ParticleAgent3D : public Hal::AgentPT3D<ParticleGrid3D>
{
public:
    explicit ParticleAgent3D(ParticleGrid3D *pGrid)
        : Hal::AgentPT3D<ParticleGrid3D>(pGrid),
          m_tag(0)
    {
    }

    ~ParticleAgent3D()
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

class ParticleGrid3D : public Hal::AgentGrid3D<ParticleAgent3D, ParticleGrid3D>
{
public:
    ParticleGrid3D(int32_t xDim,
                   int32_t yDim,
                   int32_t zDim,
                   bool    wrapX,
                   bool    wrapY,
                   bool    wrapZ)
        : Hal::AgentGrid3D<ParticleAgent3D, ParticleGrid3D>(xDim, yDim, zDim, wrapX, wrapY, wrapZ)
    {
    }

    ~ParticleGrid3D()
    {
    }
};

TEST_CASE("AgentPT3D: multiple agents can stack on the same voxel and relink correctly on Dispose")
{
    ParticleGrid3D grid(10, 10, 10, false, false, false);
    ParticleAgent3D &a = grid.NewAgentPT(3.5, 3.5, 3.5);
    ParticleAgent3D &b = grid.NewAgentPT(3.2, 3.9, 3.1);
    a.SetTag(1);
    b.SetTag(2);

    CHECK(a.GetCountOnSquare() == 2);
    std::vector<ParticleAgent3D *> onVoxel;
    grid.GetAgents(onVoxel, 3, 3, 3);
    REQUIRE(onVoxel.size() == 2);
    CHECK(onVoxel[0] == &b);
    CHECK(onVoxel[1] == &a);

    b.Dispose();
    CHECK(a.GetCountOnSquare() == 1);
    onVoxel.clear();
    grid.GetAgents(onVoxel, 3, 3, 3);
    REQUIRE(onVoxel.size() == 1);
    CHECK(onVoxel[0] == &a);
}

TEST_CASE("AgentGrid3D::GetAgentsRad matches a brute-force distance check over all agents, with wraparound")
{
    ParticleGrid3D grid(15, 15, 15, true, true, true);
    Hal::Rand rand(99);

    const int32_t agentCount = 150;
    std::vector<ParticleAgent3D *> allAgents;
    for (int32_t i = 0; i < agentCount; i++)
    {
        double x = rand.Double(15.0);
        double y = rand.Double(15.0);
        double z = rand.Double(15.0);
        ParticleAgent3D &a = grid.NewAgentPT(x, y, z);
        a.SetTag(i);
        allAgents.push_back(&a);
    }

    struct Query
    {
        double x;
        double y;
        double z;
        double rad;
    };
    std::vector<Query> queries = {
        {7.5, 7.5, 7.5, 3.0},
        {0.2, 0.3, 0.1, 4.0},
        {14.8, 14.7, 14.9, 5.0}};

    for (const Query &q : queries)
    {
        std::vector<ParticleAgent3D *> found;
        grid.GetAgentsRad(found, q.x, q.y, q.z, q.rad);

        std::vector<int32_t> foundTags;
        for (ParticleAgent3D *pAgent : found)
        {
            foundTags.push_back(pAgent->Tag());
        }
        std::sort(foundTags.begin(), foundTags.end());

        std::vector<int32_t> expectedTags;
        for (ParticleAgent3D *pAgent : allAgents)
        {
            double dx = grid.DispX(q.x, pAgent->Xpt());
            double dy = grid.DispY(q.y, pAgent->Ypt());
            double dz = grid.DispZ(q.z, pAgent->Zpt());
            if (std::sqrt(dx * dx + dy * dy + dz * dz) < q.rad)
            {
                expectedTags.push_back(pAgent->Tag());
            }
        }
        std::sort(expectedTags.begin(), expectedTags.end());

        CAPTURE(q.x);
        CAPTURE(q.y);
        CAPTURE(q.z);
        CHECK(foundTags == expectedTags);
        CHECK_FALSE(expectedTags.empty());
    }
}

TEST_CASE("AgentGrid3D::GetAgentsRad EvalAgent and displacement-info overloads (wrap-aware)")
{
    ParticleGrid3D grid(15, 15, 15, true, true, true);
    Hal::Rand rand(4242);

    const int32_t agentCount = 150;
    std::vector<ParticleAgent3D *> allAgents;
    for (int32_t i = 0; i < agentCount; i++)
    {
        ParticleAgent3D &a = grid.NewAgentPT(rand.Double(15.0), rand.Double(15.0), rand.Double(15.0));
        a.SetTag(i);
        allAgents.push_back(&a);
    }

    // Query near the wrapped corner so displacements take the short way.
    const double queryX = 0.4;
    const double queryY = 14.6;
    const double queryZ = 0.2;
    const double rad = 4.0;
    auto evenTag = [](ParticleAgent3D *pAgent)
    {
        return (pAgent->Tag() % 2) == 0;
    };

    std::vector<ParticleAgent3D *> filtered;
    grid.GetAgentsRad(filtered, queryX, queryY, queryZ, rad, evenTag);

    std::vector<int32_t> foundTags;
    for (ParticleAgent3D *pAgent : filtered)
    {
        foundTags.push_back(pAgent->Tag());
    }
    std::sort(foundTags.begin(), foundTags.end());

    std::vector<int32_t> expectedTags;
    for (ParticleAgent3D *pAgent : allAgents)
    {
        double dx = grid.DispX(queryX, pAgent->Xpt());
        double dy = grid.DispY(queryY, pAgent->Ypt());
        double dz = grid.DispZ(queryZ, pAgent->Zpt());
        if ((std::sqrt(dx * dx + dy * dy + dz * dz) < rad) && ((pAgent->Tag() % 2) == 0))
        {
            expectedTags.push_back(pAgent->Tag());
        }
    }
    std::sort(expectedTags.begin(), expectedTags.end());

    CHECK(foundTags == expectedTags);
    CHECK_FALSE(expectedTags.empty());

    // Displacement-info overload: entries align with the found list and
    // hold the wrap-aware displacement/distance.
    std::vector<ParticleAgent3D *> found;
    std::vector<Hal::AgentDisplacementInfo3D> info;
    grid.GetAgentsRad(found, info, queryX, queryY, queryZ, rad);
    REQUIRE(info.size() == found.size());
    CHECK_FALSE(found.empty());
    for (size_t k = 0; k < found.size(); k++)
    {
        ParticleAgent3D *pAgent = found[k];
        double dx = grid.DispX(queryX, pAgent->Xpt());
        double dy = grid.DispY(queryY, pAgent->Ypt());
        double dz = grid.DispZ(queryZ, pAgent->Zpt());
        CHECK(info[k].dispX == doctest::Approx(dx));
        CHECK(info[k].dispY == doctest::Approx(dy));
        CHECK(info[k].dispZ == doctest::Approx(dz));
        CHECK(info[k].dist == doctest::Approx(std::sqrt(dx * dx + dy * dy + dz * dz)));
        CHECK(info[k].dist < rad);
    }
}
