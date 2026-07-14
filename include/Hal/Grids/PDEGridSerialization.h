#pragma once

#include "Hal/Grids/PDEGrid1D.h"
#include "Hal/Grids/PDEGrid2D.h"
#include "Hal/Grids/PDEGrid3D.h"
#include "Hal/IO/BinaryReader.h"
#include "Hal/IO/BinaryWriter.h"

namespace Hal
{
    // Free-function Save/Load pairs for the PDE grid family. Unlike
    // GridSerialization.h's
    // value grids, PDEGrid also persists the not-yet-committed deltas
    // buffer and the update counter, via each class's SetState() —
    // without deltas, a save taken between Set()/Add() calls and the
    // next Update() would silently lose those pending changes on load.
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
