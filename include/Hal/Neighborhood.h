#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Each function below returns a neighborhood-offset table centered on
    // the origin. The layout is a leading run of zero-padding (as many
    // slots as there are neighbors) followed by the actual (x,y[,z])
    // offset pairs/triples. The padding exists because HAL's Agent
    // MapHood() reuses the front of this same array as scratch space to
    // write out computed neighbor grid-indices during a neighborhood
    // query, rather than allocating a second array — that convention is
    // carried over unchanged from the Java version since Phase 2's agent
    // neighborhood code depends on this exact layout.

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
