#include <doctest/doctest.h>

#include "Hal/Grids/PDEGrid1D.h"
#include "Hal/Grids/PDEGrid2D.h"
#include "Hal/Grids/PDEGrid3D.h"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------
// The explicit diffusion stencil has an exact closed-form effect on a
// periodic sinusoidal mode: for field[x] = sin(2*pi*k*x/N), one step of
// Diffusion(diffRate) scales the mode by exactly
// lambda = 1 - 4*diffRate*sin(pi*k/N)^2 (from
// sin(a+b)+sin(a-b) = 2*sin(a)*cos(b)). This gives a ground-truth check
// of the stencil's correctness — not just "the error shrinks with
// refinement" — and is the discrete analogue of the analytic
// eigenfunction decay HAL's own ConvergenceTesting harness probes.
// ---------------------------------------------------------------------

TEST_CASE("PDEGrid1D::Diffusion exactly matches the analytic eigenmode decay factor (periodic)")
{
    const int32_t n = 32;
    const double diffRate = 0.2;
    Hal::PDEGrid1D grid(n, true);

    for (int32_t x = 0; x < n; x++)
    {
        grid.Set(x, std::sin(2.0 * M_PI * x / n));
    }
    grid.Update();

    grid.Diffusion(diffRate);
    grid.Update();

    double lambda = 1.0 - 4.0 * diffRate * std::pow(std::sin(M_PI / n), 2);
    for (int32_t x = 0; x < n; x++)
    {
        double expected = lambda * std::sin(2.0 * M_PI * x / n);
        CAPTURE(x);
        CHECK(grid.Get(x) == doctest::Approx(expected).epsilon(1e-9));
    }
}

TEST_CASE("PDEGrid2D::Diffusion exactly matches the analytic eigenmode decay factor (periodic, product mode)")
{
    const int32_t nx = 16;
    const int32_t ny = 20;
    const double diffRate = 0.05;
    Hal::PDEGrid2D grid(nx, ny, true, true);

    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            grid.Set(x, y, std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny));
        }
    }
    grid.Update();

    grid.Diffusion(diffRate);
    grid.Update();

    double lambda = 1.0 - 4.0 * diffRate * (std::pow(std::sin(M_PI / nx), 2) + std::pow(std::sin(M_PI / ny), 2));
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            double expected = lambda * std::sin(2.0 * M_PI * x / nx) * std::sin(2.0 * M_PI * y / ny);
            CAPTURE(x);
            CAPTURE(y);
            CHECK(grid.Get(x, y) == doctest::Approx(expected).epsilon(1e-9));
        }
    }
}

TEST_CASE("PDEGrid1D::Diffusion conserves total mass under both periodic and zero-flux boundaries")
{
    const int32_t n = 10;
    for (bool wrap : {true, false})
    {
        Hal::PDEGrid1D grid(n, wrap);
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

        for (int32_t step = 0; step < 20; step++)
        {
            grid.Diffusion(0.4);
            grid.Update();
        }

        double after = 0;
        for (int32_t x = 0; x < n; x++)
        {
            after += grid.Get(x);
        }
        CAPTURE(wrap);
        CHECK(after == doctest::Approx(before));
    }
}

TEST_CASE("PDEGrid2D::Diffusion and Update throw on unstable rates/argument mismatches")
{
    Hal::PDEGrid2D grid(5, 5);
    CHECK_THROWS_AS(grid.Diffusion(0.26), std::invalid_argument);
    CHECK_NOTHROW(grid.Diffusion(0.25));
    CHECK_THROWS_AS(grid.Advection(0.3, 0.3), std::invalid_argument);
    CHECK_NOTHROW(grid.Advection(0.25, 0.25));
}

TEST_CASE("PDEGrid1D::Diffusion/Advection stability thresholds match HAL's documented limits")
{
    Hal::PDEGrid1D grid(5);
    CHECK_THROWS_AS(grid.Diffusion(0.51), std::invalid_argument);
    CHECK_NOTHROW(grid.Diffusion(0.5));
    CHECK_THROWS_AS(grid.Advection(0.51), std::invalid_argument);
    CHECK_NOTHROW(grid.Advection(0.5));
}

TEST_CASE("PDEGrid3D::Diffusion/Advection stability thresholds match HAL's documented limits")
{
    Hal::PDEGrid3D grid(4, 4, 4);
    CHECK_THROWS_AS(grid.Diffusion(1.0 / 6.0 + 0.001), std::invalid_argument);
    CHECK_NOTHROW(grid.Diffusion(1.0 / 6.0));
    CHECK_THROWS_AS(grid.Advection(0.2, 0.2, 0.2), std::invalid_argument);
    CHECK_NOTHROW(grid.Advection(0.2, 0.15, 0.15));
}

TEST_CASE("PDEGrid1D: Set/Add/Mul/Scale accumulate into deltas, visible only after Update")
{
    Hal::PDEGrid1D grid(4);
    grid.Set(0, 5.0);
    CHECK(grid.Get(0) == 0.0);
    grid.Update();
    CHECK(grid.Get(0) == doctest::Approx(5.0));

    grid.Add(0, 2.0);
    grid.Update();
    CHECK(grid.Get(0) == doctest::Approx(7.0));

    grid.Mul(0, 0.5);
    grid.Update();
    CHECK(grid.Get(0) == doctest::Approx(10.5));

    grid.Scale(0, 2.0);
    grid.Update();
    CHECK(grid.Get(0) == doctest::Approx(21.0));
}

TEST_CASE("PDEGrid1D::Advection shifts a periodic pulse by one cell per unit CFL step")
{
    const int32_t n = 8;
    Hal::PDEGrid1D grid(n, true);
    grid.Set(0, 1.0);
    grid.Update();

    // A pure upwind step at vel=1 would move the pulse by one full cell,
    // but vel=1 exceeds the 0.5 stability bound, so this only checks the
    // stable, partial-shift case: after one step at vel=0.5, half the
    // pulse should have moved into the downwind neighbor.
    grid.Advection(0.5);
    grid.Update();

    CHECK(grid.Get(0) == doctest::Approx(0.5));
    CHECK(grid.Get(1) == doctest::Approx(0.5));

    double total = 0;
    for (int32_t x = 0; x < n; x++)
    {
        total += grid.Get(x);
    }
    CHECK(total == doctest::Approx(1.0));
}

TEST_CASE("PDEGrid1D::Diffusion with a constant Dirichlet boundary leaves a matching-value flat field unchanged")
{
    Hal::PDEGrid1D grid(6, false);
    grid.SetAll(3.0);
    grid.Update();

    grid.Diffusion(0.4, 3.0);
    grid.Update();

    for (int32_t x = 0; x < 6; x++)
    {
        CAPTURE(x);
        CHECK(grid.Get(x) == doctest::Approx(3.0));
    }
}

// ---------------------------------------------------------------------
// Gradient methods (central difference of the committed field), including
// regression coverage for two Java bugs the port fixes: PDEGrid2D's
// GradientY wrapped Y with the X-axis flag, and PDEGrid3D's GradientZ
// bounds-checked the Y dimension instead of Z.
// ---------------------------------------------------------------------

TEST_CASE("PDEGrid2D gradients: central difference of a linear field, interior")
{
    Hal::PDEGrid2D grid(6, 5);
    for (int32_t x = 0; x < 6; x++)
    {
        for (int32_t y = 0; y < 5; y++)
        {
            grid.Set(x, y, 3.0 * x + 7.0 * y);
        }
    }
    grid.Update();

    // f = 3x + 7y -> central diff over 2 cells is 2*3 in x, 2*7 in y.
    CHECK(grid.GradientX(2, 2) == doctest::Approx(6.0));
    CHECK(grid.GradientY(2, 2) == doctest::Approx(14.0));
}

TEST_CASE("PDEGrid2D gradients: reflecting boundary off-grid, and Dirichlet override")
{
    Hal::PDEGrid2D grid(4, 4);
    for (int32_t x = 0; x < 4; x++)
    {
        for (int32_t y = 0; y < 4; y++)
        {
            grid.Set(x, y, static_cast<double>(x));
        }
    }
    grid.Update();

    // At x=0 (no wrap): left reflects to field[0]=0, right=field[1]=1.
    CHECK(grid.GradientX(0, 1) == doctest::Approx(1.0));
    // Dirichlet: left boundary value 10 -> right(=1) - 10 = -9.
    CHECK(grid.GradientX(0, 1, 10.0) == doctest::Approx(1.0 - 10.0));
}

TEST_CASE("PDEGrid2D GradientY wraps along Y, not X (regression for the Java wrap bug)")
{
    // Wrap only in Y, not X. A field varying in y so the wrap direction is
    // observable: Java's GradientY used the X wrap flag, which here is
    // false, so it would reflect instead of wrap.
    Hal::PDEGrid2D grid(4, 4, false, true);
    for (int32_t x = 0; x < 4; x++)
    {
        for (int32_t y = 0; y < 4; y++)
        {
            grid.Set(x, y, static_cast<double>(y));
        }
    }
    grid.Update();

    // At y=0 with Y-wrap: down wraps to field[y=3]=3, up=field[y=1]=1 -> 1-3.
    CHECK(grid.GradientY(1, 0) == doctest::Approx(1.0 - 3.0));
}

TEST_CASE("PDEGrid3D gradients: central difference of a linear field, interior")
{
    Hal::PDEGrid3D grid(4, 4, 4);
    for (int32_t x = 0; x < 4; x++)
    {
        for (int32_t y = 0; y < 4; y++)
        {
            for (int32_t z = 0; z < 4; z++)
            {
                grid.Set(x, y, z, 100.0 * x + 10.0 * y + z);
            }
        }
    }
    grid.Update();

    CHECK(grid.GradientX(1, 1, 1) == doctest::Approx(200.0));
    CHECK(grid.GradientY(1, 1, 1) == doctest::Approx(20.0));
    CHECK(grid.GradientZ(1, 1, 1) == doctest::Approx(2.0));
}

TEST_CASE("PDEGrid3D GradientZ uses the Z axis at the boundary (regression for the Java dimension bug)")
{
    // f = 100x + 10y + z. Java's GradientZ displaced z via a Y-axis check,
    // so at z=0 it read neighboring cells (x,y+/-1) instead of applying the
    // z boundary — a wrong result. The correct reflecting value is field
    // at z=0 itself.
    Hal::PDEGrid3D grid(4, 4, 4);
    for (int32_t x = 0; x < 4; x++)
    {
        for (int32_t y = 0; y < 4; y++)
        {
            for (int32_t z = 0; z < 4; z++)
            {
                grid.Set(x, y, z, 100.0 * x + 10.0 * y + z);
            }
        }
    }
    grid.Update();

    // z=0, no wrap: down reflects to f(1,1,0), up = f(1,1,1) -> 1.
    CHECK(grid.GradientZ(1, 1, 0) == doctest::Approx(1.0));
    // z=3 (top), no wrap: up reflects to f(1,1,3), down = f(1,1,2) -> 1.
    CHECK(grid.GradientZ(1, 1, 3) == doctest::Approx(1.0));
}

TEST_CASE("PDEGrid2D discontinuous diffusion with uniform rates matches the constant-rate stencil")
{
    const int32_t nx = 8;
    const int32_t ny = 6;
    const double rate = 0.15;

    // Reference: homogeneous diffusion.
    Hal::PDEGrid2D reference(nx, ny, true, false);
    // Discontinuous with uniform rate arrays.
    Hal::PDEGrid2D varying(nx, ny, true, false);
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            double v = std::sin(2.0 * M_PI * x / nx) + 0.5 * y;
            reference.Set(x, y, v);
            varying.Set(x, y, v);
        }
    }
    reference.Update();
    varying.Update();

    std::vector<double> ratesX(static_cast<size_t>(nx * ny), rate);
    std::vector<double> ratesY(static_cast<size_t>(nx * ny), rate);

    reference.Diffusion(rate);
    varying.Diffusion(ratesX, ratesY);
    reference.Update();
    varying.Update();

    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            CAPTURE(x);
            CAPTURE(y);
            CHECK(varying.Get(x, y) == doctest::Approx(reference.Get(x, y)).epsilon(1e-12));
        }
    }
}

TEST_CASE("PDEGrid2D discontinuous diffusion: zero rate in a region blocks flux there")
{
    const int32_t n = 6;
    Hal::PDEGrid2D grid(n, n);
    // A single hot cell in the middle.
    grid.Set(3, 3, 1.0);
    grid.Update();

    // Zero diffusion everywhere -> nothing moves.
    std::vector<double> zeroRates(static_cast<size_t>(n * n), 0.0);
    grid.Diffusion(zeroRates, zeroRates);
    grid.Update();

    for (int32_t x = 0; x < n; x++)
    {
        for (int32_t y = 0; y < n; y++)
        {
            CAPTURE(x);
            CAPTURE(y);
            double expected = ((x == 3) && (y == 3)) ? 1.0 : 0.0;
            CHECK(grid.Get(x, y) == doctest::Approx(expected));
        }
    }
}

TEST_CASE("PDEGrid3D discontinuous diffusion with uniform rates matches the constant-rate stencil")
{
    const int32_t nx = 5;
    const int32_t ny = 4;
    const int32_t nz = 3;
    const double rate = 0.1;

    Hal::PDEGrid3D reference(nx, ny, nz, true, false, true);
    Hal::PDEGrid3D varying(nx, ny, nz, true, false, true);
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            for (int32_t z = 0; z < nz; z++)
            {
                double v = std::sin(2.0 * M_PI * x / nx) + 0.5 * y + 0.25 * z;
                reference.Set(x, y, z, v);
                varying.Set(x, y, z, v);
            }
        }
    }
    reference.Update();
    varying.Update();

    std::vector<double> rates(static_cast<size_t>(nx * ny * nz), rate);
    reference.Diffusion(rate);
    varying.Diffusion(rates, rates, rates);
    reference.Update();
    varying.Update();

    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            for (int32_t z = 0; z < nz; z++)
            {
                CAPTURE(x);
                CAPTURE(y);
                CAPTURE(z);
                CHECK(varying.Get(x, y, z) == doctest::Approx(reference.Get(x, y, z)).epsilon(1e-12));
            }
        }
    }
}

TEST_CASE("PDEGrid3D discontinuous diffusion: a zero-rate slab blocks flux across it")
{
    const int32_t n = 5;
    Hal::PDEGrid3D grid(n, n, n);
    grid.Set(2, 2, 2, 1.0);
    grid.Update();

    // Each z-interface's rate is stored at the lower-index cell, so the two
    // interfaces bounding the hot cell's z=2 layer are governed by ratesZ at
    // z==1 (the 1|2 face) and z==2 (the 2|3 face). Zeroing both isolates the
    // hot cell in z while leaving x/y flux free to move.
    std::vector<double> ratesX(static_cast<size_t>(n * n * n), 0.05);
    std::vector<double> ratesY(static_cast<size_t>(n * n * n), 0.05);
    std::vector<double> ratesZ(static_cast<size_t>(n * n * n), 0.05);
    for (int32_t x = 0; x < n; x++)
    {
        for (int32_t y = 0; y < n; y++)
        {
            ratesZ[(x * n + y) * n + 1] = 0.0;
            ratesZ[(x * n + y) * n + 2] = 0.0;
        }
    }

    grid.Diffusion(ratesX, ratesY, ratesZ);
    grid.Update();

    // No mass leaked to the z-neighbors of the hot cell.
    CHECK(grid.Get(2, 2, 1) == doctest::Approx(0.0));
    CHECK(grid.Get(2, 2, 3) == doctest::Approx(0.0));
    // But x/y neighbors did receive flux.
    CHECK(grid.Get(1, 2, 2) > 0.0);
    CHECK(grid.Get(2, 1, 2) > 0.0);
}

// ---------------------------------------------------------------------
// Per-cell velocity-field advection (the conservative flux form — see
// PDEequations.h). With a wrapped domain and constant velocity arrays,
// the flux form telescopes to exactly the uniform-velocity upwind
// scheme, so the two overloads must agree cell-for-cell.

TEST_CASE("PDEGrid1D per-cell-velocity advection with a uniform field matches the constant-velocity scheme (periodic)")
{
    const int32_t n = 16;
    const double  vel = 0.3;

    Hal::PDEGrid1D uniformGrid(n, true);
    Hal::PDEGrid1D perCellGrid(n, true);
    for (int32_t x = 0; x < n; x++)
    {
        double value = std::sin(2.0 * 3.14159265358979323846 * x / n) + 2.0;
        uniformGrid.Set(x, value);
        perCellGrid.Set(x, value);
    }
    uniformGrid.Update();
    perCellGrid.Update();

    std::vector<double> vels(static_cast<size_t>(n), vel);
    uniformGrid.Advection(vel);
    perCellGrid.Advection(vels);
    uniformGrid.Update();
    perCellGrid.Update();

    for (int32_t x = 0; x < n; x++)
    {
        CHECK(perCellGrid.Get(x) == doctest::Approx(uniformGrid.Get(x)).epsilon(1e-12));
    }
}

TEST_CASE("PDEGrid2D per-cell-velocity advection matches the constant-velocity scheme (periodic) and conserves mass with a varying field")
{
    const int32_t n = 12;

    Hal::PDEGrid2D uniformGrid(n, n, true, true);
    Hal::PDEGrid2D perCellGrid(n, n, true, true);
    for (int32_t x = 0; x < n; x++)
    {
        for (int32_t y = 0; y < n; y++)
        {
            double value = std::sin(2.0 * 3.14159265358979323846 * (x + 2 * y) / n) + 2.0;
            uniformGrid.Set(x, y, value);
            perCellGrid.Set(x, y, value);
        }
    }
    uniformGrid.Update();
    perCellGrid.Update();

    std::vector<double> xVels(static_cast<size_t>(n * n), 0.2);
    std::vector<double> yVels(static_cast<size_t>(n * n), -0.15);
    uniformGrid.Advection(0.2, -0.15);
    perCellGrid.Advection(xVels, yVels);
    uniformGrid.Update();
    perCellGrid.Update();

    for (int32_t i = 0; i < n * n; i++)
    {
        CHECK(perCellGrid.Get(i) == doctest::Approx(uniformGrid.Get(i)).epsilon(1e-12));
    }

    // Now a genuinely varying (rotational) velocity field on the wrapped
    // domain: the flux form must conserve total mass exactly.
    double massBefore = perCellGrid.GetAvg() * n * n;
    for (int32_t x = 0; x < n; x++)
    {
        for (int32_t y = 0; y < n; y++)
        {
            int32_t i = x * n + y;
            xVels[static_cast<size_t>(i)] = 0.2 * std::sin(2.0 * 3.14159265358979323846 * y / n);
            yVels[static_cast<size_t>(i)] = 0.2 * std::cos(2.0 * 3.14159265358979323846 * x / n);
        }
    }
    perCellGrid.Advection(xVels, yVels);
    perCellGrid.Update();
    double massAfter = perCellGrid.GetAvg() * n * n;
    CHECK(massAfter == doctest::Approx(massBefore).epsilon(1e-12));
}

TEST_CASE("PDEGrid2D per-cell-velocity advection conserves mass with zero-flux walls")
{
    const int32_t n = 10;

    Hal::PDEGrid2D grid(n, n, false, false);
    std::vector<double> xVels(static_cast<size_t>(n * n));
    std::vector<double> yVels(static_cast<size_t>(n * n));
    for (int32_t x = 0; x < n; x++)
    {
        for (int32_t y = 0; y < n; y++)
        {
            int32_t i = x * n + y;
            grid.Set(x, y, 1.0 + 0.1 * x + 0.05 * y);
            xVels[static_cast<size_t>(i)] = 0.15 * std::sin(0.9 * x + 0.3 * y);
            yVels[static_cast<size_t>(i)] = 0.15 * std::cos(0.4 * x - 0.7 * y);
        }
    }
    grid.Update();

    // The off-grid velocity is 0 at every wall (zero-flux), so even a
    // messy interior velocity field can't create or destroy mass.
    double massBefore = grid.GetAvg() * n * n;
    grid.Advection(xVels, yVels);
    grid.Update();
    double massAfter = grid.GetAvg() * n * n;
    CHECK(massAfter == doctest::Approx(massBefore).epsilon(1e-12));
}

TEST_CASE("PDEGrid1D per-cell-velocity advection transports a pulse one cell downstream")
{
    const int32_t n = 8;
    const double  vel = 0.25;

    Hal::PDEGrid1D grid(n, true);
    grid.Set(3, 1.0);
    grid.Update();

    std::vector<double> vels(static_cast<size_t>(n), vel);
    grid.Advection(vels);
    grid.Update();

    // Upwind flux form, uniform positive velocity: the pulse cell loses
    // vel * value out its +x face, the downstream neighbor gains it.
    CHECK(grid.Get(3) == doctest::Approx(1.0 - vel));
    CHECK(grid.Get(4) == doctest::Approx(vel));
    CHECK(grid.Get(2) == doctest::Approx(0.0));
    CHECK(grid.Get(5) == doctest::Approx(0.0));
}

TEST_CASE("PDEGrid3D per-cell z-velocity advection matches per-pencil 1D advection (regression for the Java z-axis bugs)")
{
    // Velocities purely along z, varying with z, on a non-wrapping grid:
    // each (x, y) pencil must evolve exactly like the 1D per-cell scheme
    // applied to that pencil. Java's per-cell Advection3 fails this: its
    // z lookups went through the Y-axis helper (mis-indexing at both z
    // boundaries — out of bounds at the last cell), read the z-minus
    // velocity from the center cell, and wrapped the z edge by the x
    // axis's flag (docs/JAVA_BUGS_FOUND.md #18).
    const int32_t nx = 3;
    const int32_t ny = 3;
    const int32_t nz = 6;

    Hal::PDEGrid3D grid(nx, ny, nz, false, false, false);
    std::vector<double> zPencilField(static_cast<size_t>(nz));
    std::vector<double> zPencilVels(static_cast<size_t>(nz));
    for (int32_t z = 0; z < nz; z++)
    {
        zPencilField[static_cast<size_t>(z)] = 1.0 + 0.5 * z;
        // Negative velocity at the far z end exercises the exact lookup
        // that read past the end of the array in Java.
        zPencilVels[static_cast<size_t>(z)] = (z >= nz / 2) ? -0.2 : 0.15;
    }

    std::vector<double> xVels(static_cast<size_t>(nx * ny * nz), 0.0);
    std::vector<double> yVels(static_cast<size_t>(nx * ny * nz), 0.0);
    std::vector<double> zVels(static_cast<size_t>(nx * ny * nz));
    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            for (int32_t z = 0; z < nz; z++)
            {
                grid.Set(x, y, z, zPencilField[static_cast<size_t>(z)]);
                zVels[static_cast<size_t>(x * ny * nz + y * nz + z)] = zPencilVels[static_cast<size_t>(z)];
            }
        }
    }
    grid.Update();
    grid.Advection(xVels, yVels, zVels);
    grid.Update();

    // 1D reference on one pencil.
    Hal::PDEGrid1D pencil(nz, false);
    for (int32_t z = 0; z < nz; z++)
    {
        pencil.Set(z, zPencilField[static_cast<size_t>(z)]);
    }
    pencil.Update();
    pencil.Advection(zPencilVels);
    pencil.Update();

    for (int32_t x = 0; x < nx; x++)
    {
        for (int32_t y = 0; y < ny; y++)
        {
            for (int32_t z = 0; z < nz; z++)
            {
                CHECK(grid.Get(x, y, z) == doctest::Approx(pencil.Get(z)).epsilon(1e-12));
            }
        }
    }
}

TEST_CASE("PDEGrid1D per-cell-velocity advection with Dirichlet boundaries admits inflow through the wall")
{
    const int32_t n = 3;
    const double  vel = 0.2;
    const double  boundaryVal = 5.0;

    Hal::PDEGrid1D grid(n, false);
    grid.Update();

    // Empty field, uniform +x velocity, Dirichlet field value 5 outside:
    // cell 0's -x face pulls in vel * boundaryVal; every other flux moves
    // zeros around. Cell 2 loses vel * field[2] = 0 out the +x wall.
    std::vector<double> vels(static_cast<size_t>(n), vel);
    auto fieldBC = [boundaryVal](int32_t)
    {
        return boundaryVal;
    };
    auto velBC = [vel](int32_t)
    {
        return vel;
    };
    grid.Advection(vels, fieldBC, velBC);
    grid.Update();

    CHECK(grid.Get(0) == doctest::Approx(vel * boundaryVal));
    CHECK(grid.Get(1) == doctest::Approx(0.0));
    CHECK(grid.Get(2) == doctest::Approx(0.0));
}
