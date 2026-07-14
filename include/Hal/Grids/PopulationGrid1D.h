#pragma once

#include "Hal/Grids/GridGeometry1D.h"
#include "Hal/Grids/PopulationGridBase.h"

#include <cstdint>

namespace Hal
{
    // Mean-field/count-based 1D population grid — see PopulationGrid2D.h
    // for the rationale behind the TCount template and the deferred
    // Diffusion() method.
    template <typename TCount>
    class PopulationGrid1D : public GridGeometry1D, public PopulationGridBase<TCount>
    {
    public:
        PopulationGrid1D(int32_t xDim)
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

        // NOTE: unlike the 2D/3D versions, Java's PopulationGrid1D adds no
        // coordinate-based Get/Set/Add/Move overloads of its own — in 1D
        // the x coordinate already *is* the flat index, so the inherited
        // index-based methods (brought into scope below) are the whole
        // API surface.
        using PopulationGridBase<TCount>::Get;
        using PopulationGridBase<TCount>::Set;
        using PopulationGridBase<TCount>::Add;
        using PopulationGridBase<TCount>::Move;
    };
}
