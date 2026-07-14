#include <doctest/doctest.h>

#include "Hal/Tools/OdeSolver.h"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------
// Ported from Tools/ODESolver/ODESolver.java. Verified against exact
// closed-form solutions rather than eyeballed trajectories: dy/dt = y has
// solution e^t; the linear system (y0'=y1, y1'=-y0) with y0(0)=1,y1(0)=0
// has solution (cos t, -sin t). These give a real ground truth for the
// integrators' accuracy and order.
// ---------------------------------------------------------------------

static void ExpDerivative(double /*t*/, const double *pState, double *pOut)
{
    pOut[0] = pState[0];
}

// Simple harmonic oscillator: y0' = y1, y1' = -y0.
static void OscillatorDerivative(double /*t*/, const double *pState, double *pOut)
{
    pOut[0] = pState[1];
    pOut[1] = -pState[0];
}

TEST_CASE("OdeSolver::Runge4 integrates dy/dt = y to e^t with high accuracy")
{
    Hal::OdeSolver solver;
    std::vector<double> state = {1.0};
    solver.Runge4Integrate(ExpDerivative, state.data(), 1, 0.0, 1.0, 0.001);
    CHECK(state[0] == doctest::Approx(std::exp(1.0)).epsilon(1e-9));
}

TEST_CASE("OdeSolver::Runge4 exhibits ~4th-order convergence (halving dt cuts error ~16x)")
{
    Hal::OdeSolver solver;
    double target = std::exp(1.0);

    std::vector<double> coarse = {1.0};
    solver.Runge4Integrate(ExpDerivative, coarse.data(), 1, 0.0, 1.0, 0.02);
    double errorCoarse = std::abs(coarse[0] - target);

    std::vector<double> fine = {1.0};
    solver.Runge4Integrate(ExpDerivative, fine.data(), 1, 0.0, 1.0, 0.01);
    double errorFine = std::abs(fine[0] - target);

    double ratio = errorCoarse / errorFine;
    // 4th order => ~16x; allow a generous band around it.
    CHECK(ratio > 8.0);
    CHECK(ratio < 32.0);
}

TEST_CASE("OdeSolver::Euler is first-order and far less accurate than Runge4 for the same step")
{
    Hal::OdeSolver solver;
    double target = std::exp(1.0);

    std::vector<double> euler = {1.0};
    solver.EulerIntegrate(ExpDerivative, euler.data(), 1, 0.0, 1.0, 0.001);

    std::vector<double> runge = {1.0};
    solver.Runge4Integrate(ExpDerivative, runge.data(), 1, 0.0, 1.0, 0.001);

    double eulerError = std::abs(euler[0] - target);
    double rungeError = std::abs(runge[0] - target);
    CHECK(eulerError > rungeError);
    // Euler with dt=0.001 lands in the right ballpark but visibly off.
    CHECK(euler[0] == doctest::Approx(target).epsilon(1e-2));
}

TEST_CASE("OdeSolver::Runge45 adaptively integrates dy/dt = y to e^t within tolerance")
{
    Hal::OdeSolver solver;
    std::vector<double> state = {1.0};
    double finalDt = solver.Runge45(ExpDerivative, state.data(), state.data(), 1, 0.0, 1.0, 0.01, 1e-9);
    CHECK(state[0] == doctest::Approx(std::exp(1.0)).epsilon(1e-6));
    CHECK(finalDt > 0.0);
}

TEST_CASE("OdeSolver::Runge4 conserves the harmonic oscillator over a full period")
{
    Hal::OdeSolver solver;
    std::vector<double> state = {1.0, 0.0};
    double period = 2.0 * M_PI;
    solver.Runge4Integrate(OscillatorDerivative, state.data(), 2, 0.0, period, 0.0005);
    // After one full period: (cos 2pi, -sin 2pi) = (1, 0).
    CHECK(state[0] == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(state[1] == doctest::Approx(0.0).epsilon(1e-6));
}

// Regression test for a Java bug fixed in this port: Java's separate-out
// Runge45 overload re-evaluates the derivative at the *original* state
// array every step (rather than the running output), so it only
// integrates correctly when state and out alias — which Java's own
// convenience overloads always do, masking the bug. This port evaluates
// at the current state, so integrating in place and integrating into a
// separate output buffer must give identical results.
TEST_CASE("OdeSolver::Runge45 into a separate output buffer matches in-place integration")
{
    Hal::OdeSolver solver;

    std::vector<double> inPlace = {1.0};
    solver.Runge45(ExpDerivative, inPlace.data(), inPlace.data(), 1, 0.0, 1.0, 0.01, 1e-9);

    std::vector<double> stateIn = {1.0};
    std::vector<double> out = {0.0};
    solver.Runge45(ExpDerivative, stateIn.data(), out.data(), 1, 0.0, 1.0, 0.01, 1e-9);

    CHECK(out[0] == doctest::Approx(inPlace[0]).epsilon(1e-12));
    // The input buffer must be left untouched when integrating into out.
    CHECK(stateIn[0] == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("OdeSolver::Runge45 clamped overload rejects dtMax < dtMin")
{
    Hal::OdeSolver solver;
    std::vector<double> state = {1.0};
    CHECK_THROWS_AS(
        solver.Runge45(ExpDerivative, state.data(), state.data(), 1, 0.0, 1.0, 0.01, 1e-9, 0.1, 0.01),
        std::invalid_argument);
}

TEST_CASE("OdeSolver::Runge45 clamped overload keeps the step within [dtMin, dtMax] and still converges")
{
    Hal::OdeSolver solver;
    std::vector<double> state = {1.0};
    double finalDt = solver.Runge45(ExpDerivative, state.data(), state.data(), 1, 0.0, 1.0, 0.01, 1e-9, 0.001, 0.05);
    CHECK(state[0] == doctest::Approx(std::exp(1.0)).epsilon(1e-5));
    CHECK(finalDt <= 0.05 + 1e-12);
    CHECK(finalDt >= 0.001 - 1e-12);
}

TEST_CASE("OdeSolver::Runge4Record captures the whole trajectory, ending at the analytic value")
{
    Hal::OdeSolver solver;
    std::vector<std::vector<double>> states = {{1.0}};
    std::vector<double> ts = {0.0};

    int32_t last = solver.Runge4Record(ExpDerivative, states, ts, 1.0, 0.05, 0);

    // 1.0 / 0.05 = 20 steps recorded on top of the initial state.
    CHECK(last == 20);
    REQUIRE(states.size() == 21);
    REQUIRE(ts.size() == 21);
    CHECK(ts.front() == doctest::Approx(0.0));
    CHECK(ts.back() == doctest::Approx(1.0));
    // Each recorded state should track e^t at its time.
    for (size_t i = 0; i < states.size(); i++)
    {
        CAPTURE(i);
        CHECK(states[i][0] == doctest::Approx(std::exp(ts[i])).epsilon(1e-6));
    }
}

TEST_CASE("OdeSolver::EulerRecord grows the trajectory and stays close to e^t")
{
    Hal::OdeSolver solver;
    std::vector<std::vector<double>> states = {{1.0}};
    std::vector<double> ts = {0.0};

    solver.EulerRecord(ExpDerivative, states, ts, 1.0, 0.001, 0);
    CHECK(states.size() == 1001);
    CHECK(states.back()[0] == doctest::Approx(std::exp(1.0)).epsilon(1e-2));
}

TEST_CASE("OdeSolver::Runge45Record records a variable-step trajectory reaching the analytic value")
{
    Hal::OdeSolver solver;
    std::vector<std::vector<double>> states = {{1.0}};
    std::vector<double> ts = {0.0};

    int32_t last = solver.Runge45Record(ExpDerivative, states, ts, 1.0, 0.05, 1e-9, 0);
    CHECK(last >= 1);
    CHECK(ts.back() == doctest::Approx(1.0).epsilon(1e-9));
    CHECK(states.back()[0] == doctest::Approx(std::exp(1.0)).epsilon(1e-6));
    // Times must be strictly increasing.
    for (size_t i = 1; i < ts.size(); i++)
    {
        CAPTURE(i);
        CHECK(ts[i] > ts[i - 1]);
    }
}

TEST_CASE("OdeSolver trajectory recording rejects a missing start state")
{
    Hal::OdeSolver solver;
    std::vector<std::vector<double>> states;
    std::vector<double> ts;
    CHECK_THROWS_AS(solver.Runge4Record(ExpDerivative, states, ts, 1.0, 0.1, 0), std::invalid_argument);
}
