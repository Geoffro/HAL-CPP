#pragma once

#include "Hal/Grids/GridGeometry2D.h"
#include "Hal/Grids/PopulationGridBase.h"
#include "Hal/Tools/MultinomialCalc.h"

#include <cstdint>

namespace Hal
{
    // Mean-field/count-based 2D population grid, ported from
    // GridsAndAgents/PopulationGrid2D.java (and PopulationGridLong2D.java,
    // collapsed into this one template on TCount — see PopulationGridBase.h).
    //
    // NOTE: Diffusion() is deferred to Phase 3 (depends on MultinomialCalc
    // and PopulationGridPDEequations, neither ported yet).
    template <typename TCount>
    class PopulationGrid2D : public GridGeometry2D, public PopulationGridBase<TCount>
    {
    public:
        PopulationGrid2D(int32_t xDim,
                         int32_t yDim)
            : PopulationGrid2D(xDim, yDim, false, false)
        {
        }

        PopulationGrid2D(int32_t xDim,
                         int32_t yDim,
                         bool    wrapX,
                         bool    wrapY)
            : GridGeometry2D(xDim, yDim, wrapX, wrapY),
              PopulationGridBase<TCount>(xDim * yDim)
        {
        }

        ~PopulationGrid2D()
        {
        }

        TCount Get(int32_t x,
                   int32_t y) const
        {
            return this->Get(I(x, y));
        }

        using PopulationGridBase<TCount>::Get;

        void Set(int32_t x,
                 int32_t y,
                 TCount  val)
        {
            int32_t i = I(x, y);
            this->Add(i, val - this->Get(i));
        }

        using PopulationGridBase<TCount>::Set;

        void Add(int32_t x,
                 int32_t y,
                 TCount  val)
        {
            this->Add(I(x, y), val);
        }

        using PopulationGridBase<TCount>::Add;

        void Move(int32_t xFrom,
                  int32_t yFrom,
                  int32_t xTo,
                  int32_t yTo,
                  TCount  val)
        {
            this->Move(I(xFrom, yFrom), I(xTo, yTo), val);
        }

        using PopulationGridBase<TCount>::Move;

        // Mean-field diffusion: each occupied cell's population is split
        // multinomially among the four von-Neumann directions, each moved
        // with probability diffRate (the remainder stays put). Movement goes
        // through the delta buffer, so call Update() afterward. Zero-flux at
        // non-wrapping boundaries (those agents stay home — no sample drawn,
        // matching PopulationGridPDEequations.Diffusion2 with no boundary
        // condition). Ported from PopulationGrid2D.Diffusion.
        void Diffusion(double           diffRate,
                       MultinomialCalc &mn)
        {
            this->ApplyOccupied([&](int32_t i, TCount ct) {
                mn.Setup(static_cast<int32_t>(ct));
                int32_t x = ItoX(i);
                int32_t y = ItoY(i);
                DiffuseX(x + 1, y, i, diffRate, mn);
                DiffuseX(x - 1, y, i, diffRate, mn);
                DiffuseY(x, y + 1, i, diffRate, mn);
                DiffuseY(x, y - 1, i, diffRate, mn);
            });
        }

    private:
        void DiffuseX(int32_t          x,
                      int32_t          y,
                      int32_t          centerI,
                      double           prob,
                      MultinomialCalc &mn)
        {
            if (InDim(x, Xdim()))
            {
                TCount c = static_cast<TCount>(mn.Sample(prob));
                this->Add(centerI, -c);
                this->Add(x * Ydim() + y, c);
            }
            else if (IsWrapX())
            {
                int32_t xw = Wrap(x, Xdim());
                TCount  c = static_cast<TCount>(mn.Sample(prob));
                this->Add(centerI, -c);
                this->Add(xw * Ydim() + y, c);
            }
            // else zero-flux: the agents stay in the center cell.
        }

        void DiffuseY(int32_t          x,
                      int32_t          y,
                      int32_t          centerI,
                      double           prob,
                      MultinomialCalc &mn)
        {
            if (InDim(y, Ydim()))
            {
                TCount c = static_cast<TCount>(mn.Sample(prob));
                this->Add(centerI, -c);
                this->Add(x * Ydim() + y, c);
            }
            else if (IsWrapY())
            {
                int32_t yw = Wrap(y, Ydim());
                TCount  c = static_cast<TCount>(mn.Sample(prob));
                this->Add(centerI, -c);
                this->Add(x * Ydim() + yw, c);
            }
            // else zero-flux: the agents stay in the center cell.
        }
    };
}
