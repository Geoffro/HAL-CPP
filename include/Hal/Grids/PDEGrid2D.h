#pragma once

#include "Hal/Grids/GridGeometry2D.h"
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
    // 2D reaction-diffusion-advection grid, ported from
    // GridsAndAgents/PDEGrid2D.java. Double-buffered like a value grid
    // (Get reads the committed field, Set/Add/Mul/Scale accumulate into
    // deltas, Update() commits) — this is the same field/deltas/Update()
    // shape used throughout, and each explicit stencil cell's update
    // depends only on the read-only current field plus its own delta
    // slot, which is exactly the compute-shader-friendly structure
    // Phase 4 is expected to build on (see docs/DESIGN.md).
    //
    // NOTE: deferred from this port: the discontinuous/spatially-varying
    // diffusion-rate and velocity-field overloads (`Diffusion`/`Advection`
    // taking per-cell rate arrays or grids), and the Gradient methods.
    class PDEGrid2D : public GridGeometry2D
    {
    public:
        PDEGrid2D(int32_t xDim,
                  int32_t yDim)
            : PDEGrid2D(xDim, yDim, false, false)
        {
        }

        PDEGrid2D(int32_t xDim,
                  int32_t yDim,
                  bool    wrapX,
                  bool    wrapY)
            : GridGeometry2D(xDim, yDim, wrapX, wrapY),
              m_field(static_cast<size_t>(Length()), 0.0),
              m_deltas(static_cast<size_t>(Length()), 0.0),
              m_updateCt(0),
              m_scratch(static_cast<size_t>(Length()), 0.0),
              m_tdma(std::max(xDim, yDim))
        {
        }

        ~PDEGrid2D()
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

        double Get(int32_t x,
                   int32_t y) const
        {
            return m_field[I(x, y)];
        }

        double Get(int32_t i) const
        {
            return m_field[i];
        }

        void Set(int32_t x,
                 int32_t y,
                 double  val)
        {
            int32_t i = I(x, y);
            m_deltas[i] = val - m_field[i];
        }

        void Set(int32_t i,
                 double  val)
        {
            m_deltas[i] = val - m_field[i];
        }

        void Add(int32_t x,
                 int32_t y,
                 double  val)
        {
            m_deltas[I(x, y)] += val;
        }

        void Add(int32_t i,
                 double  val)
        {
            m_deltas[i] += val;
        }

        // Multiplies a value in the current field and adds the change to
        // the delta field.
        void Mul(int32_t x,
                 int32_t y,
                 double  val)
        {
            int32_t i = I(x, y);
            m_deltas[i] += m_field[i] * val;
        }

        void Mul(int32_t i,
                 double  val)
        {
            m_deltas[i] += m_field[i] * val;
        }

        // Scales the field value by val upon Update() (val=1 is a no-op).
        void Scale(int32_t x,
                   int32_t y,
                   double  val)
        {
            int32_t i = I(x, y);
            m_deltas[i] += m_field[i] * (val - 1);
        }

        void Scale(int32_t i,
                   double  val)
        {
            m_deltas[i] += m_field[i] * (val - 1);
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

        // Commits the delta field into the current field and clears it.
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

        // Returns the max absolute delta — call right before Update().
        double MaxDelta() const
        {
            double maxDif = 0;
            for (int32_t i = 0; i < Length(); i++)
            {
                maxDif = std::max(maxDif, std::abs(m_deltas[i]));
            }
            return maxDif;
        }

        // Like MaxDelta, but scaled relative to the current field value;
        // denomOffset avoids dividing by zero.
        double MaxDeltaScaled(double denomOffset) const
        {
            double maxDif = 0;
            for (int32_t i = 0; i < Length(); i++)
            {
                maxDif = std::max(maxDif, std::abs(m_deltas[i] / (std::abs(m_field[i]) + denomOffset)));
            }
            return maxDif;
        }

        // Clamps any cell that would go negative on the next Update() to
        // zero — call before Update().
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

        // Central-difference gradient of the committed field in the X
        // direction (field[x+1] - field[x-1], with zero-flux / reflecting
        // boundaries off-grid unless the axis wraps).
        double GradientX(int32_t x,
                         int32_t y) const
        {
            return DisplacedFieldX(x + 1, y, Get(x, y)) - DisplacedFieldX(x - 1, y, Get(x, y));
        }

        // As above, but using a fixed Dirichlet boundaryValue off-grid.
        double GradientX(int32_t x,
                         int32_t y,
                         double  boundaryValue) const
        {
            return DisplacedFieldX(x + 1, y, boundaryValue) - DisplacedFieldX(x - 1, y, boundaryValue);
        }

        double GradientY(int32_t x,
                         int32_t y) const
        {
            return DisplacedFieldY(x, y + 1, Get(x, y)) - DisplacedFieldY(x, y - 1, Get(x, y));
        }

        double GradientY(int32_t x,
                         int32_t y,
                         double  boundaryValue) const
        {
            return DisplacedFieldY(x, y + 1, boundaryValue) - DisplacedFieldY(x, y - 1, boundaryValue);
        }

        // Explicit diffusion (5-point stencil). diffCoef is the
        // nondimensionalized diffusion coefficient; values above 0.25 are
        // numerically unstable for this explicit scheme.
        void Diffusion(double diffCoef)
        {
            CheckDiffusionStable(diffCoef);
            Internal::Diffusion2(m_field.data(), m_deltas.data(), diffCoef, Xdim(), Ydim(), IsWrapX(), IsWrapY());
        }

        // Same, but with a constant Dirichlet boundary value instead of
        // zero flux.
        void Diffusion(double diffCoef,
                       double boundaryValue)
        {
            CheckDiffusionStable(diffCoef);
            Internal::Diffusion2(m_field.data(), m_deltas.data(), diffCoef, Xdim(), Ydim(), IsWrapX(), IsWrapY(),
                                  [boundaryValue](int32_t, int32_t)
                                  { return boundaryValue; });
        }

        // Same, with a general boundary-condition callable(x,y) -> value.
        template <typename BCFn>
        void Diffusion(double diffCoef,
                       BCFn   bc)
        {
            CheckDiffusionStable(diffCoef);
            Internal::Diffusion2(m_field.data(), m_deltas.data(), diffCoef, Xdim(), Ydim(), IsWrapX(), IsWrapY(), bc);
        }

        // Diffusion with a spatially-varying (discontinuous) rate:
        // diffRatesX / diffRatesY give the per-cell rate for x- and
        // y-direction flux (each Length()-sized, row-major like the field).
        // Zero-flux boundaries. Caller is responsible for keeping the rates
        // within the explicit-scheme stability limit.
        void Diffusion(const std::vector<double> &diffRatesX,
                       const std::vector<double> &diffRatesY)
        {
            Internal::Diffusion2(m_field.data(), m_deltas.data(), diffRatesX.data(), diffRatesY.data(), Xdim(), Ydim(), IsWrapX(), IsWrapY());
        }

        // Implicit (ADI) diffusion — unconditionally stable, no CFL cap,
        // at the cost of a tridiagonal solve per row/column per step.
        void DiffusionADI(double diffRate)
        {
            Internal::Diffusion2DADI(m_field.data(), m_scratch.data(), m_deltas.data(), diffRate, Xdim(), Ydim(), IsWrapX(), IsWrapY(), m_tdma);
        }

        void DiffusionADI(double diffRate,
                          double boundaryValue)
        {
            Internal::Diffusion2DADI(m_field.data(), m_scratch.data(), m_deltas.data(), diffRate, Xdim(), Ydim(), IsWrapX(), IsWrapY(),
                                      [boundaryValue](int32_t, int32_t)
                                      { return boundaryValue; },
                                      m_tdma);
        }

        template <typename BCFn>
        void DiffusionADI(double diffRate,
                          BCFn   bc)
        {
            Internal::Diffusion2DADI(m_field.data(), m_scratch.data(), m_deltas.data(), diffRate, Xdim(), Ydim(), IsWrapX(), IsWrapY(), bc, m_tdma);
        }

        // First-order upwind advection. |xVel|+|yVel| above 0.5 is
        // numerically unstable for this explicit scheme.
        void Advection(double xVel,
                       double yVel)
        {
            CheckAdvectionStable(xVel, yVel);
            Internal::Advection2(m_field.data(), m_deltas.data(), xVel, yVel, Xdim(), Ydim(), IsWrapX(), IsWrapY());
        }

        void Advection(double xVel,
                       double yVel,
                       double boundaryValue)
        {
            CheckAdvectionStable(xVel, yVel);
            Internal::Advection2(m_field.data(), m_deltas.data(), xVel, yVel, Xdim(), Ydim(), IsWrapX(), IsWrapY(),
                                  [boundaryValue](int32_t, int32_t)
                                  { return boundaryValue; });
        }

        template <typename BCFn>
        void Advection(double xVel,
                       double yVel,
                       BCFn   bc)
        {
            CheckAdvectionStable(xVel, yVel);
            Internal::Advection2(m_field.data(), m_deltas.data(), xVel, yVel, Xdim(), Ydim(), IsWrapX(), IsWrapY(), bc);
        }

        // Advection with a spatially-varying per-cell velocity field
        // (xVels/yVels are Length()-sized, row-major like the field), in
        // conservative upwind flux form — see PDEequations.h. Zero-flux
        // boundaries: the off-grid velocity is 0, so nothing flows through
        // the walls. Caller is responsible for keeping the velocities
        // within the stability limit.
        void Advection(const std::vector<double> &xVels,
                       const std::vector<double> &yVels)
        {
            Internal::Advection2(m_field.data(), m_deltas.data(), xVels.data(), yVels.data(), Xdim(), Ydim(), IsWrapX(), IsWrapY());
        }

        // Dirichlet form: the callables give the off-grid field value and
        // velocity components.
        template <typename BCFn, typename XVelBCFn, typename YVelBCFn>
        void Advection(const std::vector<double> &xVels,
                       const std::vector<double> &yVels,
                       BCFn                       bc,
                       XVelBCFn                   xVelBC,
                       YVelBCFn                   yVelBC)
        {
            Internal::Advection2(m_field.data(), m_deltas.data(), xVels.data(), yVels.data(), Xdim(), Ydim(), IsWrapX(), IsWrapY(), bc, xVelBC, yVelBC);
        }

    private:
        // Committed-field value at (x, y) with x possibly off-grid: the
        // in-bounds value, the wrapped value if the x-axis wraps, otherwise
        // outOfBoundsValue.
        double DisplacedFieldX(int32_t x,
                               int32_t y,
                               double  outOfBoundsValue) const
        {
            if (InDim(x, Xdim()))
            {
                return Get(x, y);
            }
            if (IsWrapX())
            {
                return Get(Wrap(x, Xdim()), y);
            }
            return outOfBoundsValue;
        }

        // As DisplacedFieldX but for an off-grid y. NOTE: Java's
        // PDEGrid2D.GradientY wrapped the y access with the x-axis wrap flag
        // (a bug, see docs/JAVA_BUGS_FOUND.md); this uses IsWrapY().
        double DisplacedFieldY(int32_t x,
                               int32_t y,
                               double  outOfBoundsValue) const
        {
            if (InDim(y, Ydim()))
            {
                return Get(x, y);
            }
            if (IsWrapY())
            {
                return Get(x, Wrap(y, Ydim()));
            }
            return outOfBoundsValue;
        }

        static void CheckDiffusionStable(double diffCoef)
        {
            if (diffCoef > 0.25)
            {
                throw std::invalid_argument("Diffusion rate above stable maximum value of 0.25");
            }
        }

        static void CheckAdvectionStable(double xVel,
                                         double yVel)
        {
            if ((std::abs(xVel) + std::abs(yVel)) > 0.5)
            {
                throw std::invalid_argument("Advection rate component sum above stable maximum value of 0.5");
            }
        }

        std::vector<double> m_field;
        std::vector<double> m_deltas;
        int32_t m_updateCt;
        std::vector<double> m_scratch;
        TdmaSolver m_tdma;
    };
}
