#pragma once

#include "Hal/Grids/GridGeometry1D.h"
#include "Hal/Grids/PopulationGridBase.h"

#include <cstdint>

namespace Hal
{
    // 1D count-based population grid (see PopulationGridBase.h). In 1D the
    // x coordinate is the index, so the base class's index-based methods
    // are the whole API.
    template <typename TCount>
    class PopulationGrid1D : public GridGeometry1D, public PopulationGridBase<TCount>
    {
    public:
        explicit PopulationGrid1D(int32_t xDim)
            : PopulationGrid1D(xDim, false)
        {
        }

        PopulationGrid1D(int32_t xDim,
                         bool    wrapX)
            : GridGeometry1D(xDim, wrapX),
              PopulationGridBase<TCount>(xDim)
        {
        }

        ~PopulationGrid1D()
        {
        }

        using GridGeometry1D::Length;

        using PopulationGridBase<TCount>::Get;
        using PopulationGridBase<TCount>::Set;
        using PopulationGridBase<TCount>::Add;
        using PopulationGridBase<TCount>::Move;
    };
}
