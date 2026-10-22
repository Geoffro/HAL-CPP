#include <doctest/doctest.h>

#include "Hal/Grids/PDEGrid1D.h"
#include "Hal/Grids/PDEGrid2D.h"
#include "Hal/Grids/PDEGrid3D.h"

#include <cmath>

// ADI has an exact effect on a periodic sinusoidal mode. In 1D (and per
// axis in 2D) it is the Crank-Nicolson factor
//
//   lambda_CN(theta) = (1 - 2*r*sin(theta/2)^2) / (1 + 2*r*sin(theta/2)^2)
//
// with r = diffRate and theta = 2*pi*k/N; a 2D product mode scales by the
// product of the two axes' factors. In 3D each sweep's explicit half-step
// covers the two other axes, so with f(theta) = (4*r/3)*sin(theta/2)^2:
//
//   lambda_3D = [(1-fy-fz)/(1+fx)] * [(1-fx-fz)/(1+fy)] * [(1-fx-fy)/(1+fz)]
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

TEST_CASE("PDEGrid2D::DiffusionADI with a constant Dirichlet boundary leaves a matching-value flat field unchanged")
{
    Hal::PDEGrid2D grid(7, 5, false, false);
    grid.SetAll(4.0);
    grid.Update();

    grid.DiffusionADI(0.9, 4.0);
    grid.Update();

    for (int32_t i = 0; i < grid.Length(); i++)
    {
        CAPTURE(i);
        CHECK(grid.Get(i) == doctest::Approx(4.0));
    }
}

TEST_CASE("PDEGrid3D::DiffusionADI with a constant Dirichlet boundary leaves a matching-value flat field unchanged")
{
    Hal::PDEGrid3D grid(5, 4, 6, false, false, false);
    grid.SetAll(4.0);
    grid.Update();

    grid.DiffusionADI(0.9, 4.0);
    grid.Update();

    for (int32_t i = 0; i < grid.Length(); i++)
    {
        CAPTURE(i);
        CHECK(grid.Get(i) == doctest::Approx(4.0));
    }
}

TEST_CASE("PDEGrid2D::DiffusionADI with a Dirichlet boundary relaxes the interior to the boundary value")
{
    Hal::PDEGrid2D grid(8, 6, true, false);
    for (int32_t step = 0; step < 400; step++)
    {
        grid.DiffusionADI(1.5, 2.0);
        grid.Update();
    }
    for (int32_t i = 0; i < grid.Length(); i++)
    {
        CAPTURE(i);
        CHECK(grid.Get(i) == doctest::Approx(2.0).epsilon(1e-6));
    }
}

TEST_CASE("PDEGrid3D::DiffusionADI with a Dirichlet boundary relaxes the interior to the boundary value")
{
    Hal::PDEGrid3D grid(5, 4, 6, false, true, false);
    for (int32_t step = 0; step < 400; step++)
    {
        grid.DiffusionADI(1.5, 2.0);
        grid.Update();
    }
    for (int32_t i = 0; i < grid.Length(); i++)
    {
        CAPTURE(i);
        CHECK(grid.Get(i) == doctest::Approx(2.0).epsilon(1e-6));
    }
}

// With the left x boundary at 1 and the right at 3 (other edges matching),
// the steady state is the line from 1 to 3 between the cell faces.
static double DirichletRamp(int32_t x,
                            int32_t xDim)
{
    if (x < 0)
    {
        return 1.0;
    }
    if (x >= xDim)
    {
        return 3.0;
    }
    return 1.0 + 2.0 * (x + 0.5) / xDim;
}

TEST_CASE("DiffusionADI with different Dirichlet values on each side holds the linear steady state, and 1D relaxes to it")
{
    const int32_t xDim = 6;

    Hal::PDEGrid1D line(xDim, false);
    Hal::PDEGrid2D plane(xDim, 4, false, false);
    Hal::PDEGrid3D volume(xDim, 4, 3, false, false, false);
    for (int32_t i = 0; i < volume.Length(); i++)
    {
        volume.Set(i, DirichletRamp(volume.ItoX(i), xDim));
    }
    for (int32_t i = 0; i < plane.Length(); i++)
    {
        plane.Set(i, DirichletRamp(plane.ItoX(i), xDim));
    }
    for (int32_t x = 0; x < xDim; x++)
    {
        line.Set(x, DirichletRamp(x, xDim));
    }
    line.Update();
    plane.Update();
    volume.Update();

    line.DiffusionADI(0.9, [&](int32_t x) { return DirichletRamp(x, xDim); });
    plane.DiffusionADI(0.9, [&](int32_t x, int32_t) { return DirichletRamp(x, xDim); });
    volume.DiffusionADI(0.9, [&](int32_t x, int32_t, int32_t) { return DirichletRamp(x, xDim); });
    line.Update();
    plane.Update();
    volume.Update();

    for (int32_t x = 0; x < xDim; x++)
    {
        CAPTURE(x);
        CHECK(line.Get(x) == doctest::Approx(DirichletRamp(x, xDim)));
    }
    for (int32_t i = 0; i < plane.Length(); i++)
    {
        CAPTURE(i);
        CHECK(plane.Get(i) == doctest::Approx(DirichletRamp(plane.ItoX(i), xDim)));
    }
    for (int32_t i = 0; i < volume.Length(); i++)
    {
        CAPTURE(i);
        CHECK(volume.Get(i) == doctest::Approx(DirichletRamp(volume.ItoX(i), xDim)));
    }

    Hal::PDEGrid1D relaxing(xDim, false);
    for (int32_t step = 0; step < 400; step++)
    {
        relaxing.DiffusionADI(1.5, [&](int32_t x) { return DirichletRamp(x, xDim); });
        relaxing.Update();
    }
    for (int32_t x = 0; x < xDim; x++)
    {
        CAPTURE(x);
        CHECK(relaxing.Get(x) == doctest::Approx(DirichletRamp(x, xDim)).epsilon(1e-6));
    }
}
