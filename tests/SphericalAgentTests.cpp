#include <doctest/doctest.h>

#include "Hal/Agents/SphericalAgent2D.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/MathUtil.h"

// ---------------------------------------------------------------------
// "Spherical cow" 2D physics agents, ported from SphericalAgent2D.java.
// See AgentGridTests.cpp for why CellAgent/CellGrid need the forward-
// declare-both / define-agent-first pattern.
// ---------------------------------------------------------------------

class CellGrid;

class CellAgent : public Hal::SphericalAgent2D<CellGrid>
{
public:
    explicit CellAgent(CellGrid *pGrid)
        : Hal::SphericalAgent2D<CellGrid>(pGrid)
    {
    }

    ~CellAgent()
    {
    }
};

class CellGrid : public Hal::AgentGrid2D<CellAgent, CellGrid>
{
public:
    CellGrid(int32_t xDim,
             int32_t yDim)
        : Hal::AgentGrid2D<CellAgent, CellGrid>(xDim, yDim)
    {
    }

    ~CellGrid()
    {
    }
};

TEST_CASE("SphericalAgent2D: overlapping cells repel each other under a Hookean overlap force")
{
    CellGrid grid(20, 20);
    CellAgent &a = grid.NewAgentPT(9.0, 10.0);
    CellAgent &b = grid.NewAgentPT(11.0, 10.0);
    a.DefaultInit(2.0);
    b.DefaultInit(2.0);

    // Distance is 2, combined radius is 4, so they overlap by 2.
    double interactionRad = 10.0;
    const double springConst = 0.5;
    auto overlapFn = [&](double touchDist, CellAgent & /*other*/)
    {
        return touchDist > 0 ? touchDist * springConst : 0.0;
    };

    double sumA = a.SumForces(interactionRad, overlapFn);
    double sumB = b.SumForces(interactionRad, overlapFn);
    CHECK(sumA == doctest::Approx(1.0));
    CHECK(sumB == doctest::Approx(1.0));

    // a is to the left of b, so the repulsive force should push a's
    // velocity further left (negative) and b's further right (positive).
    CHECK(a.XVel() < 0.0);
    CHECK(b.XVel() > 0.0);
    CHECK(a.YVel() == doctest::Approx(0.0));
    CHECK(b.YVel() == doctest::Approx(0.0));

    double distBefore = Hal::Dist(a.Xpt(), a.Ypt(), b.Xpt(), b.Ypt());
    a.ForceMove();
    b.ForceMove();
    double distAfter = Hal::Dist(a.Xpt(), a.Ypt(), b.Xpt(), b.Ypt());
    CHECK(distAfter > distBefore);
}

TEST_CASE("SphericalAgent2D: non-overlapping cells outside interactionRad exert no force")
{
    CellGrid grid(20, 20);
    CellAgent &a = grid.NewAgentPT(2.0, 2.0);
    CellAgent &b = grid.NewAgentPT(18.0, 18.0);
    a.DefaultInit(1.0);
    b.DefaultInit(1.0);

    double sum = a.SumForces(3.0, [](double touchDist, CellAgent & /*other*/)
                              { return touchDist; });
    CHECK(sum == doctest::Approx(0.0));
    CHECK(a.XVel() == doctest::Approx(0.0));
    CHECK(a.YVel() == doctest::Approx(0.0));
}

TEST_CASE("SphericalAgent2D: ApplyFriction scales velocity and CapVelocity bounds its norm")
{
    CellGrid grid(10, 10);
    CellAgent &a = grid.NewAgentPT(5.0, 5.0);
    a.DefaultInit(1.0);
    a.SetVelocity(3.0, 4.0);

    a.ApplyFriction(0.5);
    CHECK(a.XVel() == doctest::Approx(1.5));
    CHECK(a.YVel() == doctest::Approx(2.0));

    a.SetVelocity(3.0, 4.0);
    a.CapVelocity(2.0);
    double norm = Hal::Norm(a.XVel(), a.YVel());
    CHECK(norm == doctest::Approx(2.0));
}

TEST_CASE("SphericalAgent2D: Divide places the daughter divRadius away and moves the parent the opposite way")
{
    CellGrid grid(20, 20);
    CellAgent &parent = grid.NewAgentPT(10.0, 10.0);
    parent.DefaultInit(1.0);

    double axis[2] = {1.0, 0.0};
    CellAgent *pChild = parent.Divide(2.0, axis, nullptr);
    REQUIRE(pChild != nullptr);

    CHECK(pChild->Xpt() == doctest::Approx(12.0));
    CHECK(pChild->Ypt() == doctest::Approx(10.0));
    CHECK(parent.Xpt() == doctest::Approx(8.0));
    CHECK(parent.Ypt() == doctest::Approx(10.0));
    CHECK(grid.Pop() == 2);
}
