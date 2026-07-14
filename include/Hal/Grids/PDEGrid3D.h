#pragma once

#include "Hal/Grids/GridGeometry3D.h"
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
    // 3D reaction-diffusion-advection grid — see PDEGrid2D.h for the
    // rationale behind the double-buffered field/deltas/Update() design
    // and what's deferred from this port.
    class PDEGrid3D : public GridGeometry3D
    {
    public:
        PDEGrid3D(int32_t xDim,
                  int32_t yDim,
                  int32_t zDim)
            : PDEGrid3D(xDim, yDim, zDim, false, false, false)
        {
        }

        PDEGrid3D(int32_t xDim,
                  int32_t yDim,
                  int32_t zDim,
                  bool    wrapX,
                  bool    wrapY,
                  bool    wrapZ)
            : GridGeometry3D(xDim, yDim, zDim, wrapX, wrapY, wrapZ),
              m_field(static_cast<size_t>(Length()), 0.0),
              m_deltas(static_cast<size_t>(Length()), 0.0),
              m_updateCt(0),
              m_scratch1(static_cast<size_t>(Length()), 0.0),
              m_scratch2(static_cast<size_t>(Length()), 0.0),
              m_tdma(std::max({xDim, yDim, zDim}))
        {
        }

        ~PDEGrid3D()
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
                   int32_t y,
                   int32_t z) const
        {
            return m_field[I(x, y, z)];
        }

        double Get(int32_t i) const
        {
            return m_field[i];
        }

        void Set(int32_t x,
                 int32_t y,
                 int32_t z,
                 double  val)
        {
            int32_t i = I(x, y, z);
            m_deltas[i] = val - m_field[i];
        }

        void Set(int32_t i,
                 double  val)
        {
            m_deltas[i] = val - m_field[i];
        }

        void Add(int32_t x,
                 int32_t y,
                 int32_t z,
                 double  val)
        {
            m_deltas[I(x, y, z)] += val;
        }

        void Add(int32_t i,
                 double  val)
        {
            m_deltas[i] += val;
        }

        void Mul(int32_t x,
                 int32_t y,
                 int32_t z,
                 double  val)
        {
            int32_t i = I(x, y, z);
            m_deltas[i] += m_field[i] * val;
        }

        void Mul(int32_t i,
                 double  val)
        {
            m_deltas[i] += m_field[i] * val;
        }

        void Scale(int32_t x,
                   int32_t y,
                   int32_t z,
                   double  val)
        {
            int32_t i = I(x, y, z);
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

        // Central-difference gradient of the committed field along each
        // axis (field[+1] - field[-1], with zero-flux / reflecting
        // boundaries off-grid unless that axis wraps), optionally with a
        // fixed Dirichlet boundaryValue off-grid.
        double GradientX(int32_t x,
                         int32_t y,
                         int32_t z) const
        {
            return DisplacedFieldX(x + 1, y, z, Get(x, y, z)) - DisplacedFieldX(x - 1, y, z, Get(x, y, z));
        }

        double GradientX(int32_t x,
                         int32_t y,
                         int32_t z,
                         double  boundaryValue) const
        {
            return DisplacedFieldX(x + 1, y, z, boundaryValue) - DisplacedFieldX(x - 1, y, z, boundaryValue);
        }

        double GradientY(int32_t x,
                         int32_t y,
                         int32_t z) const
        {
            return DisplacedFieldY(x, y + 1, z, Get(x, y, z)) - DisplacedFieldY(x, y - 1, z, Get(x, y, z));
        }

        double GradientY(int32_t x,
                         int32_t y,
                         int32_t z,
                         double  boundaryValue) const
        {
            return DisplacedFieldY(x, y + 1, z, boundaryValue) - DisplacedFieldY(x, y - 1, z, boundaryValue);
        }

        double GradientZ(int32_t x,
                         int32_t y,
                         int32_t z) const
        {
            return DisplacedFieldZ(x, y, z + 1, Get(x, y, z)) - DisplacedFieldZ(x, y, z - 1, Get(x, y, z));
        }

        double GradientZ(int32_t x,
                         int32_t y,
                         int32_t z,
                         double  boundaryValue) const
        {
            return DisplacedFieldZ(x, y, z + 1, boundaryValue) - DisplacedFieldZ(x, y, z - 1, boundaryValue);
        }

        // Explicit diffusion (7-point stencil); diffCoef above 1/6 is
        // numerically unstable for this explicit 3D scheme.
        void Diffusion(double diffCoef)
        {
            CheckDiffusionStable(diffCoef);
            Internal::Diffusion3(m_field.data(), m_deltas.data(), diffCoef, Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ());
        }

        void Diffusion(double diffCoef,
                       double boundaryValue)
        {
            CheckDiffusionStable(diffCoef);
            Internal::Diffusion3(m_field.data(), m_deltas.data(), diffCoef, Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ(),
                                  [boundaryValue](int32_t, int32_t, int32_t)
                                  { return boundaryValue; });
        }

        template <typename BCFn>
        void Diffusion(double diffCoef,
                       BCFn   bc)
        {
            CheckDiffusionStable(diffCoef);
            Internal::Diffusion3(m_field.data(), m_deltas.data(), diffCoef, Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ(), bc);
        }

        // Diffusion with a spatially-varying (discontinuous) rate:
        // diffRatesX / diffRatesY / diffRatesZ give the per-cell rate for
        // x-, y- and z-direction flux (each Length()-sized, row-major like
        // the field). Zero-flux boundaries. Caller is responsible for keeping
        // the rates within the explicit-scheme stability limit.
        void Diffusion(const std::vector<double> &diffRatesX,
                       const std::vector<double> &diffRatesY,
                       const std::vector<double> &diffRatesZ)
        {
            Internal::Diffusion3(m_field.data(), m_deltas.data(), diffRatesX.data(), diffRatesY.data(), diffRatesZ.data(),
                                 Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ());
        }

        // Implicit (ADI) diffusion — unconditionally stable, no CFL cap,
        // at the cost of a tridiagonal solve per pencil per axis sweep.
        void DiffusionADI(double diffRate)
        {
            Internal::Diffusion3DADI(m_field.data(), m_scratch1.data(), m_scratch2.data(), m_deltas.data(), diffRate,
                                      Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ(), m_tdma);
        }

        void DiffusionADI(double diffRate,
                          double boundaryValue)
        {
            Internal::Diffusion3DADI(m_field.data(), m_scratch1.data(), m_scratch2.data(), m_deltas.data(), diffRate,
                                      Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ(),
                                      [boundaryValue](int32_t, int32_t, int32_t)
                                      { return boundaryValue; },
                                      m_tdma);
        }

        template <typename BCFn>
        void DiffusionADI(double diffRate,
                          BCFn   bc)
        {
            Internal::Diffusion3DADI(m_field.data(), m_scratch1.data(), m_scratch2.data(), m_deltas.data(), diffRate,
                                      Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ(), bc, m_tdma);
        }

        // First-order upwind advection; |xVel|+|yVel|+|zVel| above 0.5 is
        // numerically unstable for this explicit scheme.
        void Advection(double xVel,
                       double yVel,
                       double zVel)
        {
            CheckAdvectionStable(xVel, yVel, zVel);
            Internal::Advection3(m_field.data(), m_deltas.data(), xVel, yVel, zVel, Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ());
        }

        void Advection(double xVel,
                       double yVel,
                       double zVel,
                       double boundaryValue)
        {
            CheckAdvectionStable(xVel, yVel, zVel);
            Internal::Advection3(m_field.data(), m_deltas.data(), xVel, yVel, zVel, Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ(),
                                  [boundaryValue](int32_t, int32_t, int32_t)
                                  { return boundaryValue; });
        }

        template <typename BCFn>
        void Advection(double xVel,
                       double yVel,
                       double zVel,
                       BCFn   bc)
        {
            CheckAdvectionStable(xVel, yVel, zVel);
            Internal::Advection3(m_field.data(), m_deltas.data(), xVel, yVel, zVel, Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ(), bc);
        }

        // Advection with a spatially-varying per-cell velocity field
        // (xVels/yVels/zVels are Length()-sized, laid out like the field),
        // in conservative upwind flux form — see PDEequations.h. Zero-flux
        // boundaries: the off-grid velocity is 0, so nothing flows through
        // the walls. Caller is responsible for keeping the velocities
        // within the stability limit.
        void Advection(const std::vector<double> &xVels,
                       const std::vector<double> &yVels,
                       const std::vector<double> &zVels)
        {
            Internal::Advection3(m_field.data(), m_deltas.data(), xVels.data(), yVels.data(), zVels.data(), Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ());
        }

        // Dirichlet form: the callables give the off-grid field value and
        // velocity components.
        template <typename BCFn, typename XVelBCFn, typename YVelBCFn, typename ZVelBCFn>
        void Advection(const std::vector<double> &xVels,
                       const std::vector<double> &yVels,
                       const std::vector<double> &zVels,
                       BCFn                       bc,
                       XVelBCFn                   xVelBC,
                       YVelBCFn                   yVelBC,
                       ZVelBCFn                   zVelBC)
        {
            Internal::Advection3(m_field.data(), m_deltas.data(), xVels.data(), yVels.data(), zVels.data(), Xdim(), Ydim(), Zdim(), IsWrapX(), IsWrapY(), IsWrapZ(), bc, xVelBC, yVelBC, zVelBC);
        }

    private:
        // Committed-field value at (x, y, z) with one coordinate possibly
        // off-grid: the in-bounds value, the wrapped value if that axis
        // wraps, otherwise outOfBoundsValue.
        double DisplacedFieldX(int32_t x,
                               int32_t y,
                               int32_t z,
                               double  outOfBoundsValue) const
        {
            if (InDim(x, Xdim()))
            {
                return Get(x, y, z);
            }
            if (IsWrapX())
            {
                return Get(Wrap(x, Xdim()), y, z);
            }
            return outOfBoundsValue;
        }

        double DisplacedFieldY(int32_t x,
                               int32_t y,
                               int32_t z,
                               double  outOfBoundsValue) const
        {
            if (InDim(y, Ydim()))
            {
                return Get(x, y, z);
            }
            if (IsWrapY())
            {
                return Get(x, Wrap(y, Ydim()), z);
            }
            return outOfBoundsValue;
        }

        // NOTE: Java's PDEGrid3D.GradientZ displaced z via DisplacedY3D,
        // which bounds-checks the Y dimension instead of Z (a bug, see
        // docs/JAVA_BUGS_FOUND.md) — this checks the Z axis correctly.
        double DisplacedFieldZ(int32_t x,
                               int32_t y,
                               int32_t z,
                               double  outOfBoundsValue) const
        {
            if (InDim(z, Zdim()))
            {
                return Get(x, y, z);
            }
            if (IsWrapZ())
            {
                return Get(x, y, Wrap(z, Zdim()));
            }
            return outOfBoundsValue;
        }

        static void CheckDiffusionStable(double diffCoef)
        {
            if (diffCoef > (1.0 / 6.0))
            {
                throw std::invalid_argument("3D diffusion rate above stable maximum value of 1/6");
            }
        }

        static void CheckAdvectionStable(double xVel,
                                         double yVel,
                                         double zVel)
        {
            if ((std::abs(xVel) + std::abs(yVel) + std::abs(zVel)) > 0.5)
            {
                throw std::invalid_argument("Advection rate component sum above stable maximum value of 0.5");
            }
        }

        std::vector<double> m_field;
        std::vector<double> m_deltas;
        int32_t m_updateCt;
        std::vector<double> m_scratch1;
        std::vector<double> m_scratch2;
        TdmaSolver m_tdma;
    };
}
