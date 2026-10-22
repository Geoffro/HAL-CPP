#include "Hal/Grids/PDEGridSerialization.h"

#include "Hal/Grids/GridSerialization.h"

namespace Hal
{
    void SavePDEGrid1D(BinaryWriter    &writer,
                       const PDEGrid1D &grid)
    {
        writer.WriteInt32(grid.Xdim());
        writer.WriteBool(grid.IsWrapX());
        writer.WriteInt32(grid.UpdateCt());
        writer.WriteArray(grid.GetField());
        writer.WriteArray(grid.GetDeltas());
    }

    PDEGrid1D LoadPDEGrid1D(BinaryReader &reader)
    {
        int32_t xDim = reader.ReadInt32();
        bool wrapX = reader.ReadBool();
        int32_t updateCt = reader.ReadInt32();
        Internal::CheckLoadedDims(reader, xDim, 1, 1, "LoadPDEGrid1D");
        PDEGrid1D grid(xDim, wrapX);
        std::vector<double> field(static_cast<size_t>(grid.Length()));
        std::vector<double> deltas(static_cast<size_t>(grid.Length()));
        reader.ReadArray(field);
        reader.ReadArray(deltas);
        Internal::CheckLoadedDims(reader, xDim, 1, 1, "LoadPDEGrid1D");
        grid.SetState(field, deltas, updateCt);
        return grid;
    }

    void SavePDEGrid2D(BinaryWriter    &writer,
                       const PDEGrid2D &grid)
    {
        writer.WriteInt32(grid.Xdim());
        writer.WriteInt32(grid.Ydim());
        writer.WriteBool(grid.IsWrapX());
        writer.WriteBool(grid.IsWrapY());
        writer.WriteInt32(grid.UpdateCt());
        writer.WriteArray(grid.GetField());
        writer.WriteArray(grid.GetDeltas());
    }

    PDEGrid2D LoadPDEGrid2D(BinaryReader &reader)
    {
        int32_t xDim = reader.ReadInt32();
        int32_t yDim = reader.ReadInt32();
        bool wrapX = reader.ReadBool();
        bool wrapY = reader.ReadBool();
        int32_t updateCt = reader.ReadInt32();
        Internal::CheckLoadedDims(reader, xDim, yDim, 1, "LoadPDEGrid2D");
        PDEGrid2D grid(xDim, yDim, wrapX, wrapY);
        std::vector<double> field(static_cast<size_t>(grid.Length()));
        std::vector<double> deltas(static_cast<size_t>(grid.Length()));
        reader.ReadArray(field);
        reader.ReadArray(deltas);
        Internal::CheckLoadedDims(reader, xDim, yDim, 1, "LoadPDEGrid2D");
        grid.SetState(field, deltas, updateCt);
        return grid;
    }

    void SavePDEGrid3D(BinaryWriter    &writer,
                       const PDEGrid3D &grid)
    {
        writer.WriteInt32(grid.Xdim());
        writer.WriteInt32(grid.Ydim());
        writer.WriteInt32(grid.Zdim());
        writer.WriteBool(grid.IsWrapX());
        writer.WriteBool(grid.IsWrapY());
        writer.WriteBool(grid.IsWrapZ());
        writer.WriteInt32(grid.UpdateCt());
        writer.WriteArray(grid.GetField());
        writer.WriteArray(grid.GetDeltas());
    }

    PDEGrid3D LoadPDEGrid3D(BinaryReader &reader)
    {
        int32_t xDim = reader.ReadInt32();
        int32_t yDim = reader.ReadInt32();
        int32_t zDim = reader.ReadInt32();
        bool wrapX = reader.ReadBool();
        bool wrapY = reader.ReadBool();
        bool wrapZ = reader.ReadBool();
        int32_t updateCt = reader.ReadInt32();
        Internal::CheckLoadedDims(reader, xDim, yDim, zDim, "LoadPDEGrid3D");
        PDEGrid3D grid(xDim, yDim, zDim, wrapX, wrapY, wrapZ);
        std::vector<double> field(static_cast<size_t>(grid.Length()));
        std::vector<double> deltas(static_cast<size_t>(grid.Length()));
        reader.ReadArray(field);
        reader.ReadArray(deltas);
        Internal::CheckLoadedDims(reader, xDim, yDim, zDim, "LoadPDEGrid3D");
        grid.SetState(field, deltas, updateCt);
        return grid;
    }
}
