#include <doctest/doctest.h>

#include "Hal/Grids/PDEGrid1D.h"
#include "Hal/Grids/PDEGrid2D.h"
#include "Hal/Grids/PDEGrid3D.h"

#include <cmath>

// ---------------------------------------------------------------------
// The ADI (Crank-Nicolson) implicit solver also has an exact closed-form
// effect on a periodic sinusoidal mode.
//
//   lambda_CN(theta) = (1 - 2*r*sin(theta/2)^2) / (1 + 2*r*sin(theta/2)^2)
//
// where r = diffRate and theta = 2*pi*k/N — the standard Crank-Nicolson
// amplification factor for the 1D case (diffScaleFactor=2). For 2D
// (also diffScaleFactor=2), the directional-sweep scheme happens to make
// a separable product mode's combined eigenvalue exactly the product of
// the two per-axis lambda_CN factors: each axis's explicit half-step
// only involves the *other* axis, matching the axis it's then solved
// implicitly against, so the two passes decouple cleanly.
//
// That decoupling does NOT extend to 3D. 3D uses diffScaleFactor=3, and
// each axis's explicit half-step involves the sum of the *other two*
// axes (not just one), so the three sweeps' effects don't reduce to a
// product of three independent single-axis factors. Deriving it directly
// (with f(theta) = (4*r/3)*sin(theta/2)^2 for axes x,y,z in turn):
//
//   lambda_3D = [(1-fy-fz)/(1+fx)] * [(1-fx-fz)/(1+fy)] * [(1-fx-fy)/(1+fz)]
//
// (first sweep explicit-diffuses along y,z then implicit-solves x;
// second sweep explicit-diffuses the result along x,z then implicit-
// solves y; third sweep explicit-diffuses along x,y then implicit-solves
// z — each numerator picks up the two axes not yet implicitly resolved).
namespace TestHelpers
{
    double LambdaCN(double  diffRate,
                    int32_t n,
                    int32_t k)
    {
        double theta = 2.0 * M_PI * k / n;
        double s = std::pow(std::sin(theta / 2.0), 2);
        return (1.0 - 2.0 * diffRate * s) / (1.0 + 2.0 * diffRate * s);
    }

    double Lambda3DADI(double  diffRate,
                       int32_t nx,
                       int32_t ny,
                       int32_t nz,
                       int32_t k)
    {
        auto f = [&](int32_t n)
        {
            double theta = 2.0 * M_PI * k / n;
            return (4.0 * diffRate / 3.0) * std::pow(std::sin(theta / 2.0), 2);
        };
        double fx = f(nx);
        double fy = f(ny);
        double fz = f(nz);
        return ((1.0 - fy - fz) / (1.0 + fx)) * ((1.0 - fx - fz) / (1.0 + fy)) * ((1.0 - fx - fy) / (1.0 + fz));
    }
}

using TestHelpers::Lambda3DADI;
using TestHelpers::LambdaCN;

TEST_CASE("PDEGrid1D::DiffusionADI exactly matches the Crank-Nicolson eigenmode decay factor (periodic)")
{
    const int32_t n = 24;
    const double diffRate = 0.7;
    Hal::PDEGrid1D grid(n, true);

    for (int32_t x = 0; x < n; x++)
    {
        grid.Set(x, std::sin(2.0 * M_PI * x / n));
    }
    grid.Update();

    grid.DiffusionADI(diffRate);
    grid.Update();

    double lambda = LambdaCN(diffRate, n, 1);
    for (int32_t x = 0; x < n; x++)
    {
        double expected = lambda * std::sin(2.0 * M_PI * x / n);
        CAPTURE(x);
        CHECK(grid.Get(x) == doctest::Approx(expected).epsilon(1e-8));
    }
}

TEST_CASE("PDEGrid1D::DiffusionADI remains stable and well-behaved far above the explicit CFL limit")
{
    const int32_t n = 16;
    Hal::PDEGrid1D grid(n, true);
    for (int32_t x = 0; x < n; x++)
    {
        grid.Set(x, std::sin(2.0 * M_PI * x / n));
    }
    grid.Update();

    // diffRate = 5.0 would throw for the explicit scheme (CFL cap 0.5);
    // ADI has no such cap.
    CHECK_NOTHROW(grid.DiffusionADI(5.0));
    grid.Update();

    double lambda = LambdaCN(5.0, n, 1);
    CHECK(std::abs(lambda) < 1.0);
    for (int32_t x = 0; x < n; x++)
    {
        double expected = lambda * std::sin(2.0 * M_PI * x / n);
        CAPTURE(x);
        CHECK(grid.Get(x) == doctest::Approx(expected).epsilon(1e-8));
    }
}

TEST_CASE("PDEGrid2D::DiffusionADI eigenvalue factors as the product of the two per-axis Crank-Nicolson factors (periodic)")
{
    const int32_t nx = 12;
    const int32_t ny = 18;
    const double diffRate = 0.3;
    Hal::PDEGrid2D grid(nx, ny, true, true);

    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            grid.Set(x, y, std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny));
        }
    }
    grid.Update();

    grid.DiffusionADI(diffRate);
    grid.Update();

    double lambda = LambdaCN(diffRate, nx, 1) * LambdaCN(diffRate, ny, 1);
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            double expected = lambda * std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny);
            CAPTURE(x);
            CAPTURE(y);
            CHECK(grid.Get(x, y) == doctest::Approx(expected).epsilon(1e-7));
        }
    }
}

TEST_CASE("PDEGrid3D::DiffusionADI eigenvalue factors as the product of the three per-axis Crank-Nicolson factors (periodic)")
{
    const int32_t nx = 8;
    const int32_t ny = 10;
    const int32_t nz = 6;
    const double diffRate = 0.4;
    Hal::PDEGrid3D grid(nx, ny, nz, true, true, true);

    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            for (int32_t z = 0; z < nz; z++)
            {
                grid.Set(x, y, z,
                         std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny) * std::sin(2.0 * M_PI * z / nz));
            }
        }
    }
    grid.Update();

    grid.DiffusionADI(diffRate);
    grid.Update();

    double lambda = Lambda3DADI(diffRate, nx, ny, nz, 1);
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            for (int32_t z = 0; z < nz; z++)
            {
                double expected = lambda * std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny) * std::sin(2.0 * M_PI * z / nz);
                CAPTURE(x);
                CAPTURE(y);
                CAPTURE(z);
                CHECK(grid.Get(x, y, z) == doctest::Approx(expected).epsilon(1e-6));
            }
        }
    }
}

TEST_CASE("PDEGrid1D::DiffusionADI conserves total mass under periodic boundaries")
{
    const int32_t n = 10;
    Hal::PDEGrid1D grid(n, true);
    for (int32_t x = 0; x < n; x++)
    {
        grid.Set(x, (x == 3) ? 10.0 : 0.0);
    }
    grid.Update();

    double before = 0;
    for (int32_t x = 0; x < n; x++)
    {
        before += grid.Get(x);
    }

    for (int32_t step = 0; step < 5; step++)
    {
        grid.DiffusionADI(2.0);
        grid.Update();
    }

    double after = 0;
    for (int32_t x = 0; x < n; x++)
    {
        after += grid.Get(x);
    }
    CHECK(after == doctest::Approx(before));
}

TEST_CASE("PDEGrid1D::DiffusionADI with a constant Dirichlet boundary leaves a matching-value flat field unchanged")
{
    Hal::PDEGrid1D grid(6, false);
    grid.SetAll(4.0);
    grid.Update();

    grid.DiffusionADI(0.9, 4.0);
    grid.Update();

    for (int32_t x = 0; x < 6; x++)
    {
        CAPTURE(x);
        CHECK(grid.Get(x) == doctest::Approx(4.0));
    }
}
