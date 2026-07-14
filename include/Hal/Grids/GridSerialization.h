#pragma once

#include "Hal/Grids/Grid1D.h"
#include "Hal/Grids/Grid2D.h"
#include "Hal/Grids/Grid3D.h"
#include "Hal/IO/BinaryReader.h"
#include "Hal/IO/BinaryWriter.h"

namespace Hal
{
    // Free-function Save/Load pairs for the value grid family. Each
    // Save writes dims, wrap
    // flags, then the raw field array; each Load reconstructs a
    // matching grid and reads the field back into it. Kept outside the
    // grid classes themselves since they're generic over T and don't need
    // access to anything private beyond the existing GetField() accessor.
    template <typename T>
    void SaveGrid1D(BinaryWriter    &writer,
                    const Grid1D<T> &grid)
    {
        writer.WriteInt32(grid.Xdim());
        writer.WriteBool(grid.IsWrapX());
        writer.WriteArray(grid.GetField());
    }

    template <typename T>
    Grid1D<T> LoadGrid1D(BinaryReader &reader)
    {
        int32_t xDim = reader.ReadInt32();
        bool wrapX = reader.ReadBool();
        Grid1D<T> grid(xDim, wrapX);
        reader.ReadArray(grid.GetField());
        return grid;
    }

    template <typename T>
    void SaveGrid2D(BinaryWriter    &writer,
                    const Grid2D<T> &grid)
    {
        writer.WriteInt32(grid.Xdim());
        writer.WriteInt32(grid.Ydim());
        writer.WriteBool(grid.IsWrapX());
        writer.WriteBool(grid.IsWrapY());
        writer.WriteArray(grid.GetField());
    }

    template <typename T>
    Grid2D<T> LoadGrid2D(BinaryReader &reader)
    {
        int32_t xDim = reader.ReadInt32();
        int32_t yDim = reader.ReadInt32();
        bool wrapX = reader.ReadBool();
        bool wrapY = reader.ReadBool();
        Grid2D<T> grid(xDim, yDim, wrapX, wrapY);
        reader.ReadArray(grid.GetField());
        return grid;
    }

    template <typename T>
    void SaveGrid3D(BinaryWriter    &writer,
                    const Grid3D<T> &grid)
    {
        writer.WriteInt32(grid.Xdim());
        writer.WriteInt32(grid.Ydim());
        writer.WriteInt32(grid.Zdim());
        writer.WriteBool(grid.IsWrapX());
        writer.WriteBool(grid.IsWrapY());
        writer.WriteBool(grid.IsWrapZ());
        writer.WriteArray(grid.GetField());
    }

    template <typename T>
    Grid3D<T> LoadGrid3D(BinaryReader &reader)
    {
        int32_t xDim = reader.ReadInt32();
        int32_t yDim = reader.ReadInt32();
        int32_t zDim = reader.ReadInt32();
        bool wrapX = reader.ReadBool();
        bool wrapY = reader.ReadBool();
        bool wrapZ = reader.ReadBool();
        Grid3D<T> grid(xDim, yDim, zDim, wrapX, wrapY, wrapZ);
        reader.ReadArray(grid.GetField());
        return grid;
    }
}
