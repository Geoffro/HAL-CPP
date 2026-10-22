#pragma once

#include "Hal/Grids/Grid1D.h"
#include "Hal/Grids/Grid2D.h"
#include "Hal/Grids/Grid3D.h"
#include "Hal/IO/BinaryReader.h"
#include "Hal/IO/BinaryWriter.h"

#include <cstdint>

namespace Hal
{
    namespace Internal
    {
        // Throws std::runtime_error (after logging) if reader has failed, a
        // loaded dimension is not positive, or the grid's length would
        // overflow int32_t.
        void CheckLoadedDims(const BinaryReader &reader,
                             int32_t             xDim,
                             int32_t             yDim,
                             int32_t             zDim,
                             const char         *pWhat);
    }

    // Save/Load for the value grids: dimensions, wrap flags, then the raw
    // field. Load throws std::runtime_error on a failed read or invalid
    // dimensions.
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
        Internal::CheckLoadedDims(reader, xDim, 1, 1, "LoadGrid1D");
        Grid1D<T> grid(xDim, wrapX);
        reader.ReadArray(grid.GetField());
        Internal::CheckLoadedDims(reader, xDim, 1, 1, "LoadGrid1D");
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
        Internal::CheckLoadedDims(reader, xDim, yDim, 1, "LoadGrid2D");
        Grid2D<T> grid(xDim, yDim, wrapX, wrapY);
        reader.ReadArray(grid.GetField());
        Internal::CheckLoadedDims(reader, xDim, yDim, 1, "LoadGrid2D");
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
        Internal::CheckLoadedDims(reader, xDim, yDim, zDim, "LoadGrid3D");
        Grid3D<T> grid(xDim, yDim, zDim, wrapX, wrapY, wrapZ);
        reader.ReadArray(grid.GetField());
        Internal::CheckLoadedDims(reader, xDim, yDim, zDim, "LoadGrid3D");
        return grid;
    }
}
