#pragma once

#include "Hal/Grids/GridGeometry3D.h"
#include "Hal/Grids/PopulationGridBase.h"

#include <cstdint>

namespace Hal
{
    // Mean-field/count-based 3D population grid — see PopulationGrid2D.h
    // for the rationale behind the TCount template and the deferred
    // Diffusion() method.
    template <typename TCount>
    class PopulationGrid3D : public GridGeometry3D, public PopulationGridBase<TCount>
    {
    public:
        PopulationGrid3D(int32_t xDim,
                         int32_t yDim,
                         int32_t zDim)
            : PopulationGrid3D(xDim, yDim, zDim, false, false, false)
        {
        }

        PopulationGrid3D(int32_t xDim,
                         int32_t yDim,
                         int32_t zDim,
                         bool    wrapX,
                         bool    wrapY,
                         bool    wrapZ)
            : GridGeometry3D(xDim, yDim, zDim, wrapX, wrapY, wrapZ),
              PopulationGridBase<TCount>(xDim * yDim * zDim)
        {
        }

        ~PopulationGrid3D()
        {
        }

        TCount Get(int32_t x,
                   int32_t y,
                   int32_t z) const
        {
            return this->Get(I(x, y, z));
        }

        using PopulationGridBase<TCount>::Get;

        void Set(int32_t x,
                 int32_t y,
                 int32_t z,
                 TCount  val)
        {
            int32_t i = I(x, y, z);
            this->Add(i, val - this->Get(i));
        }

        using PopulationGridBase<TCount>::Set;

        void Add(int32_t x,
                 int32_t y,
                 int32_t z,
                 TCount  val)
        {
            this->Add(I(x, y, z), val);
        }

        using PopulationGridBase<TCount>::Add;

        void Move(int32_t xFrom,
                  int32_t yFrom,
                  int32_t zFrom,
                  int32_t xTo,
                  int32_t yTo,
                  int32_t zTo,
                  TCount  val)
        {
            this->Move(I(xFrom, yFrom, zFrom), I(xTo, yTo, zTo), val);
        }

        using PopulationGridBase<TCount>::Move;
    };
}
