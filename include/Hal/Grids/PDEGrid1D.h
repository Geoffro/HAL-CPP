#pragma once

#include "Hal/Grids/GridGeometry1D.h"
#include "Hal/MathUtil.h"
#include "Hal/Pde/ADIequations.h"
#include "Hal/Pde/PDEequations.h"
#include "Hal/Pde/TdmaSolver.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Hal
{
    // 1D reaction-diffusion-advection grid — see PDEGrid2D.h for the
    // rationale behind the double-buffered field/deltas/Update() design
    // and what's deferred from this port.
    class PDEGrid1D : public GridGeometry1D
    {
    public:
        explicit PDEGrid1D(int32_t xDim)
            : PDEGrid1D(xDim, false)
        {
        }

        PDEGrid1D(int32_t xDim,
                  bool    wrapX)
            : GridGeometry1D(xDim, wrapX),
              m_field(static_cast<size_t>(Length()), 0.0),
              m_deltas(static_cast<size_t>(Length()), 0.0),
              m_updateCt(0),
              m_tdma(xDim)
        {
        }

        ~PDEGrid1D()
        {
        }

        int32_t UpdateCt() const
        {
            return m_updateCt;
        }

        const std::vector<double> &GetField() const
        {
            return m_field;
        }

        const std::vector<double> &GetDeltas() const
        {
            return m_deltas;
        }

        // Restores field/deltas/updateCt directly (e.g. after loading a
        // saved simulation) — field and
        // deltas must each already be Length()-sized, matching this
        // grid's own dimensions.
        void SetState(const std::vector<double> &field,
                      const std::vector<double> &deltas,
                      int32_t                    updateCt)
        {
            m_field = field;
            m_deltas = deltas;
            m_updateCt = updateCt;
        }

        double Get(int32_t x) const
        {
            return m_field[x];
        }

        void Set(int32_t x,
                 double  val)
        {
            m_deltas[x] = val - m_field[x];
        }

        void Add(int32_t x,
                 double  val)
        {
            m_deltas[x] += val;
        }

        void Mul(int32_t x,
                 double  val)
        {
            m_deltas[x] += m_field[x] * val;
        }

        void Scale(int32_t x,
                   double  val)
        {
            m_deltas[x] += m_field[x] * (val - 1);
        }

        double GetMax() const
        {
            double max = std::numeric_limits<double>::lowest();
            for (int32_t i = 0; i < Length(); i++)
            {
                max = std::max(m_field[i], max);
            }
            return max;
        }

        double GetMin() const
        {
            double min = std::numeric_limits<double>::max();
            for (int32_t i = 0; i < Length(); i++)
            {
                min = std::min(m_field[i], min);
            }
            return min;
        }

        double GetAvg() const
        {
            double tot = 0;
            for (int32_t i = 0; i < Length(); i++)
            {
                tot += m_field[i];
            }
            return tot / Length();
        }

        void Update()
        {
            for (int32_t i = 0; i < Length(); i++)
            {
                m_field[i] += m_deltas[i];
            }
            std::fill(m_deltas.begin(), m_deltas.end(), 0.0);
            m_updateCt++;
        }

        void SetAll(double val)
        {
            for (int32_t i = 0; i < Length(); i++)
            {
                Set(i, val);
            }
        }

        void SetAll(const double *pVals)
        {
            for (int32_t i = 0; i < Length(); i++)
            {
                Set(i, pVals[i]);
            }
        }

        void AddAll(double val)
        {
            for (int32_t i = 0; i < Length(); i++)
            {
                Add(i, val);
            }
        }

        void MulAll(double val)
        {
            for (int32_t i = 0; i < Length(); i++)
            {
                Mul(i, val);
            }
        }

        double MaxDelta() const
        {
            double maxDif = 0;
            for (int32_t i = 0; i < Length(); i++)
            {
                maxDif = std::max(maxDif, std::abs(m_deltas[i]));
            }
            return maxDif;
        }

        double MaxDeltaScaled(double denomOffset) const
        {
            double maxDif = 0;
            for (int32_t i = 0; i < Length(); i++)
            {
                maxDif = std::max(maxDif, std::abs(m_deltas[i] / (std::abs(m_field[i]) + denomOffset)));
            }
            return maxDif;
        }

        void SetNonNegative()
        {
            for (int32_t i = 0; i < Length(); i++)
            {
                if ((m_field[i] + m_deltas[i]) < 0)
                {
                    Set(i, 0);
                }
            }
        }

        // Explicit diffusion (3-point stencil); diffCoef above 0.5 is
        // numerically unstable for this explicit 1D scheme.
        void Diffusion(double diffCoef)
        {
            CheckDiffusionStable(diffCoef);
            Internal::Diffusion1(m_field.data(), m_deltas.data(), diffCoef, Xdim(), IsWrapX());
        }

        void Diffusion(double diffCoef,
                       double boundaryValue)
        {
            CheckDiffusionStable(diffCoef);
            Internal::Diffusion1(m_field.data(), m_deltas.data(), diffCoef, Xdim(), IsWrapX(),
                                  [boundaryValue](int32_t)
                                  { return boundaryValue; });
        }

        template <typename BCFn>
        void Diffusion(double diffCoef,
                       BCFn   bc)
        {
            CheckDiffusionStable(diffCoef);
            Internal::Diffusion1(m_field.data(), m_deltas.data(), diffCoef, Xdim(), IsWrapX(), bc);
        }

        // Implicit (Crank-Nicolson) diffusion — unconditionally stable,
        // no CFL cap, at the cost of a tridiagonal solve per step.
        void DiffusionADI(double diffRate)
        {
            Internal::Diffusion1ADI(m_field.data(), m_deltas.data(), diffRate, Xdim(), IsWrapX(), m_tdma);
        }

        void DiffusionADI(double diffRate,
                          double boundaryValue)
        {
            Internal::Diffusion1ADI(m_field.data(), m_deltas.data(), diffRate, Xdim(), IsWrapX(),
                                     [boundaryValue](int32_t)
                                     { return boundaryValue; },
                                     m_tdma);
        }

        template <typename BCFn>
        void DiffusionADI(double diffRate,
                          BCFn   bc)
        {
            Internal::Diffusion1ADI(m_field.data(), m_deltas.data(), diffRate, Xdim(), IsWrapX(), bc, m_tdma);
        }

        // First-order upwind advection; |xVel| above 0.5 is numerically
        // unstable for this explicit scheme.
        void Advection(double xVel)
        {
            CheckAdvectionStable(xVel);
            Internal::Advection1(m_field.data(), m_deltas.data(), xVel, Xdim(), IsWrapX());
        }

        void Advection(double xVel,
                       double boundaryValue)
        {
            CheckAdvectionStable(xVel);
            Internal::Advection1(m_field.data(), m_deltas.data(), xVel, Xdim(), IsWrapX(),
                                  [boundaryValue](int32_t)
                                  { return boundaryValue; });
        }

        template <typename BCFn>
        void Advection(double xVel,
                       BCFn   bc)
        {
            CheckAdvectionStable(xVel);
            Internal::Advection1(m_field.data(), m_deltas.data(), xVel, Xdim(), IsWrapX(), bc);
        }

        // Advection with a spatially-varying per-cell velocity field
        // (xVels is Length()-sized), in conservative upwind flux form —
        // see PDEequations.h. Zero-flux boundaries: the off-grid velocity
        // is 0, so nothing flows through the walls. Caller is responsible
        // for keeping the velocities within the stability limit.
        void Advection(const std::vector<double> &xVels)
        {
            Internal::Advection1(m_field.data(), m_deltas.data(), xVels.data(), Xdim(), IsWrapX());
        }

        // Dirichlet form: the callables give the off-grid field value and
        // x velocity.
        template <typename BCFn, typename XVelBCFn>
        void Advection(const std::vector<double> &xVels,
                       BCFn                       bc,
                       XVelBCFn                   xVelBC)
        {
            Internal::Advection1(m_field.data(), m_deltas.data(), xVels.data(), Xdim(), IsWrapX(), bc, xVelBC);
        }

    private:
        static void CheckDiffusionStable(double diffCoef)
        {
            if (diffCoef > 0.5)
            {
                throw std::invalid_argument("Diffusion rate above stable maximum value of 0.5");
            }
        }

        static void CheckAdvectionStable(double xVel)
        {
            if (std::abs(xVel) > 0.5)
            {
                throw std::invalid_argument("Advection rate above stable maximum value of 0.5");
            }
        }

        std::vector<double> m_field;
        std::vector<double> m_deltas;
        int32_t m_updateCt;
        TdmaSolver m_tdma;
    };
}
