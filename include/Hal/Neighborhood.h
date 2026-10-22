#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Each function returns a neighborhood-offset table centered on the
    // origin: one leading zero slot per neighbor, followed by the (x,y[,z])
    // offsets. MapHood writes mapped grid indices into the leading slots,
    // so a query needs no second array.

    std::vector<int32_t> VonNeumannHood(bool includeOrigin);

    std::vector<int32_t> MooreHood(bool includeOrigin);

    // All (x,y) offsets whose Euclidean distance from the origin is <=
    // radius (a filled disc), in the same padded layout as the hoods above.
    std::vector<int32_t> CircleHood(bool   includeOrigin,
                                    double radius);

    // All (x,y) offsets in the axis-aligned rectangle [-radX,radX] x
    // [-radY,radY], in the same padded layout as the hoods above.
    std::vector<int32_t> RectangleHood(bool    includeOrigin,
                                       int32_t radX,
                                       int32_t radY);

    std::vector<int32_t> HexHoodEvenY(bool includeOrigin);

    std::vector<int32_t> HexHoodOddY(bool includeOrigin);

    std::vector<int32_t> TriangleHoodSameParity(bool includeOrigin);

    std::vector<int32_t> TriangleHoodDifParity(bool includeOrigin);

    std::vector<int32_t> VonNeumannHood3D(bool includeOrigin);

    std::vector<int32_t> SemiMooreHood3D(bool includeOrigin);

    std::vector<int32_t> MooreHood3D(bool includeOrigin);

    std::vector<int32_t> CubicHoneyHood3DEvenZ(bool includeOrigin);

    std::vector<int32_t> CubicHoneyHood3DOddZ(bool includeOrigin);

    std::vector<int32_t> DiagonalHood3D(bool includeOrigin);
}
