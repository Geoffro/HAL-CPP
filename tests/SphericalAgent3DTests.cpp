#include <doctest/doctest.h>

#include "Hal/Agents/SphericalAgent3D.h"
#include "Hal/Grids/AgentGrid3D.h"
#include "Hal/MathUtil.h"

// ---------------------------------------------------------------------
// "Spherical cow" 3D physics agents, mirroring SphericalAgentTests.cpp.
// ---------------------------------------------------------------------

class CellGrid3D;

class CellAgent3D : public Hal::SphericalAgent3D<CellGrid3D>
{
public:
    explicit CellAgent3D(CellGrid3D *pGrid)
        : Hal::SphericalAgent3D<CellGrid3D>(pGrid)
    {
    }

    ~CellAgent3D()
    {
    }
};

class CellGrid3D : public Hal::AgentGrid3D<CellAgent3D, CellGrid3D>
{
public:
    CellGrid3D(int32_t xDim,
               int32_t yDim,
               int32_t zDim)
        : Hal::AgentGrid3D<CellAgent3D, CellGrid3D>(xDim, yDim, zDim)
    {
    }

    ~CellGrid3D()
    {
    }
};

TEST_CASE("SphericalAgent3D: overlapping cells repel each other under a Hookean overlap force")
{
    CellGrid3D grid(20, 20, 20);
    CellAgent3D &a = grid.NewAgentPT(9.0, 10.0, 10.0);
    CellAgent3D &b = grid.NewAgentPT(11.0, 10.0, 10.0);
    a.DefaultInit(2.0);
    b.DefaultInit(2.0);

    double interactionRad = 10.0;
    const double springConst = 0.5;
    auto overlapFn = [&](double touchDist, CellAgent3D & /*other*/)
    {
        return touchDist > 0 ? touchDist * springConst : 0.0;
    };

    double sumA = a.SumForces(interactionRad, overlapFn);
    double sumB = b.SumForces(interactionRad, overlapFn);
    CHECK(sumA == doctest::Approx(1.0));
    CHECK(sumB == doctest::Approx(1.0));

    CHECK(a.XVel() < 0.0);
    CHECK(b.XVel() > 0.0);
    CHECK(a.YVel() == doctest::Approx(0.0));
    CHECK(a.ZVel() == doctest::Approx(0.0));

    double distBefore = Hal::Dist(a.Xpt(), a.Ypt(), a.Zpt(), b.Xpt(), b.Ypt(), b.Zpt());
    a.ForceMove();
    b.ForceMove();
    double distAfter = Hal::Dist(a.Xpt(), a.Ypt(), a.Zpt(), b.Xpt(), b.Ypt(), b.Zpt());
    CHECK(distAfter > distBefore);
}

TEST_CASE("SphericalAgent3D: ApplyFriction scales velocity and CapVelocity bounds its norm")
{
    CellGrid3D grid(10, 10, 10);
    CellAgent3D &a = grid.NewAgentPT(5.0, 5.0, 5.0);
    a.DefaultInit(1.0);
    a.SetVelocity(3.0, 4.0, 0.0);

    a.ApplyFriction(0.5);
    CHECK(a.XVel() == doctest::Approx(1.5));
    CHECK(a.YVel() == doctest::Approx(2.0));

    a.SetVelocity(0.0, 3.0, 4.0);
    a.CapVelocity(2.0);
    double norm = Hal::Norm(a.XVel(), a.YVel(), a.ZVel());
    CHECK(norm == doctest::Approx(2.0));
}

TEST_CASE("SphericalAgent3D: Divide places the daughter divRadius away and moves the parent the opposite way")
{
    CellGrid3D grid(20, 20, 20);
    CellAgent3D &parent = grid.NewAgentPT(10.0, 10.0, 10.0);
    parent.DefaultInit(1.0);

    double axis[3] = {0.0, 1.0, 0.0};
    CellAgent3D *pChild = parent.Divide(2.0, axis, nullptr);
    REQUIRE(pChild != nullptr);

    CHECK(pChild->Xpt() == doctest::Approx(10.0));
    CHECK(pChild->Ypt() == doctest::Approx(12.0));
    CHECK(pChild->Zpt() == doctest::Approx(10.0));
    CHECK(parent.Xpt() == doctest::Approx(10.0));
    CHECK(parent.Ypt() == doctest::Approx(8.0));
    CHECK(parent.Zpt() == doctest::Approx(10.0));
    CHECK(grid.Pop() == 2);
}
