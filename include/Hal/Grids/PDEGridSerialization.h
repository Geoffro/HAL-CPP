#pragma once

#include "Hal/Grids/PDEGrid1D.h"
#include "Hal/Grids/PDEGrid2D.h"
#include "Hal/Grids/PDEGrid3D.h"
#include "Hal/IO/BinaryReader.h"
#include "Hal/IO/BinaryWriter.h"

namespace Hal
{
    // Save/Load for the PDE grids: dimensions, wrap flags, update count,
    // field, and pending deltas (so a save between Add() and Update() loses
    // nothing). Load throws std::runtime_error on a failed read or invalid
    // dimensions.
    void SavePDEGrid1D(BinaryWriter    &writer,
                       const PDEGrid1D &grid);
    PDEGrid1D LoadPDEGrid1D(BinaryReader &reader);

    void SavePDEGrid2D(BinaryWriter    &writer,
                       const PDEGrid2D &grid);
    PDEGrid2D LoadPDEGrid2D(BinaryReader &reader);

    void SavePDEGrid3D(BinaryWriter    &writer,
                       const PDEGrid3D &grid);
    PDEGrid3D LoadPDEGrid3D(BinaryReader &reader);
}
