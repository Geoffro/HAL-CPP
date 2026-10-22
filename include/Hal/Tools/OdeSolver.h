#pragma once

#include "Hal/MathUtil.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Hal
{
    // ODE integrator for dy/dt = f(t, y) over a flat state vector. The
    // derivative is a callable
    //
    //     derivative(double t, const double *pState, double *pDerivativesOut)
    //
    // writing `length` values. Euler and classical Runge-Kutta 4 are
    // fixed-step; Runge45 (Runge-Kutta-Fehlberg 4(5)) adapts its step.
    // The *Record variants append every step to caller-owned state/time
    // vectors. Scratch buffers are reused across calls.
    class OdeSolver
    {
    public:
        OdeSolver()
            : m_scratch{}
        {
        }

        ~OdeSolver()
        {
        }

        OdeSolver(const OdeSolver &other) = delete;
        OdeSolver &operator=(const OdeSolver &other) = delete;

        // One Euler step of size dt from time t: pOut = pState + dt*f.
        // pState and pOut may alias (pass the same pointer to step in
        // place).
        template <typename DerivativeFn>
        void Euler(DerivativeFn  derivative,
                   const double *pState,
                   double       *pOut,
                   int32_t       length,
                   double        t,
                   double        dt)
        {
            SetupScratch(length, 1);
            derivative(t, pState, m_scratch[0].data());
            EulerAdd(pState, m_scratch[0].data(), dt, pOut, length);
        }

        // Euler integration from t0 to tf in steps of dt, in place.
        template <typename DerivativeFn>
        void EulerIntegrate(DerivativeFn derivative,
                            double      *pState,
                            int32_t      length,
                            double       t0,
                            double       tf,
                            double       dt)
        {
            double t = t0;
            while (t < tf)
            {
                if ((t + dt) > tf)
                {
                    dt = tf - t;
                }
                Euler(derivative, pState, pState, length, t, dt);
                t += dt;
            }
        }

        // One classical Runge-Kutta 4 step of size dt from time t.
        // pState and pOut may alias.
        template <typename DerivativeFn>
        void Runge4(DerivativeFn  derivative,
                    const double *pState,
                    double       *pOut,
                    int32_t       length,
                    double        t,
                    double        dt)
        {
            double halfStep = dt / 2.0;
            SetupScratch(length, 5);
            double *pK1 = m_scratch[1].data();
            double *pK2 = m_scratch[2].data();
            double *pK3 = m_scratch[3].data();
            double *pK4 = m_scratch[4].data();
            double *pTmp = m_scratch[0].data();

            derivative(t, pState, pK1);
            EulerAdd(pState, pK1, halfStep, pTmp, length);
            derivative(t + halfStep, pTmp, pK2);
            EulerAdd(pState, pK2, halfStep, pTmp, length);
            derivative(t + halfStep, pTmp, pK3);
            EulerAdd(pState, pK3, dt, pTmp, length);
            derivative(t + dt, pTmp, pK4);
            for (int32_t i = 0; i < length; i++)
            {
                pOut[i] = pState[i] + (pK1[i] + 2 * pK2[i] + 2 * pK3[i] + pK4[i]) * dt / 6.0;
            }
        }

        // Runge-Kutta 4 integration from t0 to tf in steps of dt, in place.
        template <typename DerivativeFn>
        void Runge4Integrate(DerivativeFn derivative,
                             double      *pState,
                             int32_t      length,
                             double       t0,
                             double       tf,
                             double       dt)
        {
            double t = t0;
            while (t < tf)
            {
                if ((t + dt) > tf)
                {
                    dt = tf - t;
                }
                Runge4(derivative, pState, pState, length, t, dt);
                t += dt;
            }
        }

        // Adaptive Runge-Kutta-Fehlberg 4(5) integration from t0 to tf,
        // writing the result into pOut (which may alias pState). dtStart is
        // the initial step size; the step is scaled up/down to keep the
        // per-step error near errorTolerance. Returns the final step size
        // reached (useful as the dtStart for a continued integration).
        template <typename DerivativeFn>
        double Runge45(DerivativeFn  derivative,
                       const double *pState,
                       double       *pOut,
                       int32_t       length,
                       double        t0,
                       double        tf,
                       double        dtStart,
                       double        errorTolerance)
        {
            bool first = true;
            double dt = dtStart;
            double t = t0;
            while (t < tf)
            {
                if (dt > (tf - t))
                {
                    dt = tf - t;
                }
                double error = Runge45Internal(derivative, first ? pState : pOut, length, t, dt);
                double scale = std::pow(0.84 * (errorTolerance / error), 1.0 / 4);
                if (error > errorTolerance)
                {
                    dt *= scale;
                    error = Runge45Internal(derivative, first ? pState : pOut, length, t, dt);
                    scale = std::pow(0.84 * (errorTolerance / error), 1.0 / 4);
                }
                const double *pBase = first ? pState : pOut;
                for (int32_t i = 0; i < length; i++)
                {
                    pOut[i] = pBase[i] + m_scratch[7][i];
                }
                first = false;
                t += dt;
                dt *= scale;
            }
            return dt;
        }

        // As above, but clamping the adaptive step to [dtMin, dtMax].
        template <typename DerivativeFn>
        double Runge45(DerivativeFn  derivative,
                       const double *pState,
                       double       *pOut,
                       int32_t       length,
                       double        t0,
                       double        tf,
                       double        dtStart,
                       double        errorTolerance,
                       double        dtMin,
                       double        dtMax)
        {
            if (dtMax < dtMin)
            {
                throw std::invalid_argument("OdeSolver::Runge45 dtMax can't be less than dtMin");
            }
            bool first = true;
            double dt = dtStart;
            double t = t0;
            while (t < tf)
            {
                if (dt > (tf - t))
                {
                    dt = tf - t;
                }
                double error = Runge45Internal(derivative, first ? pState : pOut, length, t, dt);
                double scale = std::pow(0.84 * (errorTolerance / error), 1.0 / 4);
                if ((dt > dtMin) && (error > errorTolerance))
                {
                    dt *= scale;
                    if (dt < dtMin)
                    {
                        dt = dtMin;
                    }
                    error = Runge45Internal(derivative, first ? pState : pOut, length, t, dt);
                    scale = std::pow(0.84 * (errorTolerance / error), 1.0 / 4);
                }
                const double *pBase = first ? pState : pOut;
                for (int32_t i = 0; i < length; i++)
                {
                    pOut[i] = pBase[i] + m_scratch[7][i];
                }
                first = false;
                t += dt;
                dt *= scale;
                if (dt > dtMax)
                {
                    dt = dtMax;
                }
                else if (dt < dtMin)
                {
                    dt = dtMin;
                }
            }
            return dt;
        }

        // Fixed-step Euler integration that records the whole trajectory:
        // starting from states[startStateIndex] at time ts[startStateIndex],
        // it appends each successive state and its time to the states / ts
        // vectors (growing them), and returns the index of the final
        // recorded state. Useful for plotting the solution over time.
        template <typename DerivativeFn>
        int32_t EulerRecord(DerivativeFn                      derivative,
                            std::vector<std::vector<double>> &states,
                            std::vector<double>              &ts,
                            double                            tf,
                            double                            dt,
                            int32_t                           startStateIndex)
        {
            RequireStartState(states, ts, startStateIndex);
            int32_t iState = startStateIndex;
            int32_t length = static_cast<int32_t>(states[static_cast<size_t>(iState)].size());
            double t = ts[static_cast<size_t>(iState)];
            while (t < tf)
            {
                if ((t + dt) > tf)
                {
                    dt = tf - t;
                }
                iState++;
                EnsureRecordSlot(states, ts, iState, length);
                Euler(derivative, states[static_cast<size_t>(iState - 1)].data(), states[static_cast<size_t>(iState)].data(), length, t, dt);
                t += dt;
                ts[static_cast<size_t>(iState)] = t;
            }
            return iState;
        }

        // As EulerRecord, but stepping with classical Runge-Kutta 4.
        template <typename DerivativeFn>
        int32_t Runge4Record(DerivativeFn                      derivative,
                             std::vector<std::vector<double>> &states,
                             std::vector<double>              &ts,
                             double                            tf,
                             double                            dt,
                             int32_t                           startStateIndex)
        {
            RequireStartState(states, ts, startStateIndex);
            int32_t iState = startStateIndex;
            int32_t length = static_cast<int32_t>(states[static_cast<size_t>(iState)].size());
            double t = ts[static_cast<size_t>(iState)];
            while (t < tf)
            {
                if ((t + dt) > tf)
                {
                    dt = tf - t;
                }
                iState++;
                EnsureRecordSlot(states, ts, iState, length);
                Runge4(derivative, states[static_cast<size_t>(iState - 1)].data(), states[static_cast<size_t>(iState)].data(), length, t, dt);
                t += dt;
                ts[static_cast<size_t>(iState)] = t;
            }
            return iState;
        }

        // Adaptive Runge-Kutta-Fehlberg 4(5) integration that records the
        // whole (variable-step) trajectory, appending each accepted state
        // and its time. dtStart is the initial step; the step adapts to keep
        // the per-step error near errorTolerance. Returns the final index.
        template <typename DerivativeFn>
        int32_t Runge45Record(DerivativeFn                      derivative,
                              std::vector<std::vector<double>> &states,
                              std::vector<double>              &ts,
                              double                            tf,
                              double                            dtStart,
                              double                            errorTolerance,
                              int32_t                           startStateIndex)
        {
            RequireStartState(states, ts, startStateIndex);
            int32_t iState = startStateIndex;
            int32_t length = static_cast<int32_t>(states[static_cast<size_t>(iState)].size());
            double dt = dtStart;
            double t = ts[static_cast<size_t>(iState)];
            while (t < tf)
            {
                if (dt > (tf - t))
                {
                    dt = tf - t;
                }
                double error = Runge45Internal(derivative, states[static_cast<size_t>(iState)].data(), length, t, dt);
                double scale = std::pow(0.84 * (errorTolerance / error), 1.0 / 4);
                if (error > errorTolerance)
                {
                    dt *= scale;
                    error = Runge45Internal(derivative, states[static_cast<size_t>(iState)].data(), length, t, dt);
                    scale = std::pow(0.84 * (errorTolerance / error), 1.0 / 4);
                }
                iState++;
                EnsureRecordSlot(states, ts, iState, length);
                for (int32_t i = 0; i < length; i++)
                {
                    states[static_cast<size_t>(iState)][static_cast<size_t>(i)] = states[static_cast<size_t>(iState - 1)][static_cast<size_t>(i)] + m_scratch[7][static_cast<size_t>(i)];
                }
                t += dt;
                ts[static_cast<size_t>(iState)] = t;
                dt *= scale;
            }
            return iState;
        }

    private:
        static void RequireStartState(std::vector<std::vector<double>> &states,
                                      std::vector<double>              &ts,
                                      int32_t                           startStateIndex)
        {
            if ((static_cast<int32_t>(states.size()) <= startStateIndex) || (static_cast<int32_t>(ts.size()) <= startStateIndex))
            {
                throw std::invalid_argument("OdeSolver trajectory recording requires a starting state/time at startStateIndex");
            }
        }

        static void EnsureRecordSlot(std::vector<std::vector<double>> &states,
                                     std::vector<double>              &ts,
                                     int32_t                           iState,
                                     int32_t                           length)
        {
            if (static_cast<int32_t>(states.size()) <= iState)
            {
                states.push_back(std::vector<double>(static_cast<size_t>(length), 0.0));
            }
            if (static_cast<int32_t>(ts.size()) <= iState)
            {
                ts.push_back(0.0);
            }
        }

        // One RKF45 step from time t over dt, evaluating the six stages,
        // leaving the RK5 increment in scratch[7] (added to the state by
        // the caller) and returning the per-unit-step error estimate
        // (Norm of the RK5-minus-RK4 difference, divided by dt).
        template <typename DerivativeFn>
        double Runge45Internal(DerivativeFn  derivative,
                               const double *pState,
                               int32_t       length,
                               double        t,
                               double        dt)
        {
            SetupScratch(length, 9);
            double *pS0 = m_scratch[0].data();
            double *pS1 = m_scratch[1].data();
            double *pS2 = m_scratch[2].data();
            double *pS3 = m_scratch[3].data();
            double *pS4 = m_scratch[4].data();
            double *pS5 = m_scratch[5].data();
            double *pS6 = m_scratch[6].data();
            double *pS7 = m_scratch[7].data();
            double *pS8 = m_scratch[8].data();

            SetDy1(derivative, pState, length, t, dt);

            CopyState(pState, pS0, length);
            TempAdd(pS1, 1.0 / 4, length);
            SetDy(derivative, pS2, t + dt * 1.0 / 4, dt, length);

            CopyState(pState, pS0, length);
            TempAdd(pS1, 3.0 / 32, length);
            TempAdd(pS2, 9.0 / 32, length);
            SetDy(derivative, pS3, t + dt * 3.0 / 8, dt, length);

            CopyState(pState, pS0, length);
            TempAdd(pS1, 1932.0 / 2197, length);
            TempAdd(pS2, -7200.0 / 2197, length);
            TempAdd(pS3, 7296.0 / 2197, length);
            SetDy(derivative, pS4, t + dt * 12.0 / 13, dt, length);

            CopyState(pState, pS0, length);
            TempAdd(pS1, 439.0 / 216, length);
            TempAdd(pS2, -8.0, length);
            TempAdd(pS3, 3680.0 / 513, length);
            TempAdd(pS4, -845.0 / 4104, length);
            SetDy(derivative, pS5, t + dt, dt, length);

            CopyState(pState, pS0, length);
            TempAdd(pS1, -8.0 / 27, length);
            TempAdd(pS2, 2.0, length);
            TempAdd(pS3, -3544.0 / 2565, length);
            TempAdd(pS4, 1859.0 / 4104, length);
            TempAdd(pS5, -11.0 / 40, length);
            SetDy(derivative, pS6, t + dt / 2.0, dt, length);

            for (int32_t i = 0; i < length; i++)
            {
                pS0[i] = pState[i] + 25.0 / 216 * pS1[i] + 1408.0 / 2565 * pS3[i] + 2197.0 / 4104 * pS4[i] - 1.0 / 5 * pS5[i];
                pS7[i] = 16.0 / 135 * pS1[i] + 6656.0 / 12825 * pS3[i] + 28561.0 / 56430 * pS4[i] - 9.0 / 50 * pS5[i] + 2.0 / 55 * pS6[i];
            }
            for (int32_t i = 0; i < length; i++)
            {
                pS8[i] = (pS7[i] + pState[i]) - pS0[i];
            }
            return (1.0 / dt) * Norm(pS8, length);
        }

        // Evaluates the derivative at (t, pState) into scratch[1], scaled
        // by dt (the first RKF45 stage).
        template <typename DerivativeFn>
        void SetDy1(DerivativeFn  derivative,
                    const double *pState,
                    int32_t       length,
                    double        t,
                    double        dt)
        {
            double *pS1 = m_scratch[1].data();
            derivative(t, pState, pS1);
            for (int32_t i = 0; i < length; i++)
            {
                pS1[i] *= dt;
            }
        }

        // Evaluates the derivative at (t, scratch[0]) into pK, scaled by dt
        // (the remaining RKF45 stages, whose input state has been
        // accumulated into scratch[0] by TempAdd).
        template <typename DerivativeFn>
        void SetDy(DerivativeFn derivative,
                   double      *pK,
                   double       t,
                   double       dt,
                   int32_t      length)
        {
            derivative(t, m_scratch[0].data(), pK);
            for (int32_t i = 0; i < length; i++)
            {
                pK[i] *= dt;
            }
        }

        void SetupScratch(int32_t length,
                          int32_t num);

        void EulerAdd(const double *pState,
                      const double *pDy,
                      double        dt,
                      double       *pOut,
                      int32_t       length);

        void TempAdd(const double *pA,
                     double        scale,
                     int32_t       length);

        void CopyState(const double *pState,
                       double       *pOut,
                       int32_t       length);

        std::vector<std::vector<double>> m_scratch;
    };
}
