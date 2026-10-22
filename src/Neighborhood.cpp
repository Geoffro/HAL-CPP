#include "Hal/Neighborhood.h"

#include <cmath>

namespace Hal
{
    std::vector<int32_t> VonNeumannHood(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0,
                                         0, 0, 1, 0, -1, 0, 0, 1, 0, -1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0, 0,
                                         1, 0, -1, 0, 0, 1, 0, -1};
        }
    }

    std::vector<int32_t> MooreHood(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0,
                                         1, 0,
                                         -1, 0,
                                         0, 1,
                                         0, -1,
                                         1, 1,
                                         1, -1,
                                         -1, 1,
                                         -1, -1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0,
                                         1, 0,
                                         -1, 0,
                                         0, 1,
                                         0, -1,
                                         1, 1,
                                         1, -1,
                                         -1, 1,
                                         -1, -1};
        }
    }

    std::vector<int32_t> CircleHood(bool   includeOrigin,
                                    double radius)
    {
        // Origin first if requested, then the scanned offsets; the padding
        // slots are prepended last.
        double  distSq = radius * radius;
        int32_t min = static_cast<int32_t>(std::floor(-radius));
        int32_t max = static_cast<int32_t>(std::ceil(radius));

        std::vector<int32_t> pairs;
        if (includeOrigin)
        {
            pairs.push_back(0);
            pairs.push_back(0);
        }
        for (int32_t x = min; x <= max; x++)
        {
            for (int32_t y = min; y <= max; y++)
            {
                if ((x * x + y * y) <= distSq)
                {
                    if ((x == 0) && (y == 0))
                    {
                        continue;
                    }
                    pairs.push_back(x);
                    pairs.push_back(y);
                }
            }
        }

        int32_t              ct = static_cast<int32_t>(pairs.size()) / 2;
        std::vector<int32_t> ret(static_cast<size_t>(ct), 0);
        ret.insert(ret.end(), pairs.begin(), pairs.end());
        return ret;
    }

    std::vector<int32_t> RectangleHood(bool    includeOrigin,
                                       int32_t radX,
                                       int32_t radY)
    {
        std::vector<int32_t> pairs;
        if (includeOrigin)
        {
            pairs.push_back(0);
            pairs.push_back(0);
        }
        for (int32_t x = -radX; x <= radX; x++)
        {
            for (int32_t y = -radY; y <= radY; y++)
            {
                if ((x == 0) && (y == 0))
                {
                    continue;
                }
                pairs.push_back(x);
                pairs.push_back(y);
            }
        }

        int32_t              ct = static_cast<int32_t>(pairs.size()) / 2;
        std::vector<int32_t> ret(static_cast<size_t>(ct), 0);
        ret.insert(ret.end(), pairs.begin(), pairs.end());
        return ret;
    }

    std::vector<int32_t> HexHoodEvenY(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 1, 1, 1, 0, 1, -1, 0, -1, -1, 0, 0, 1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0,
                                         1, 1, 1, 0, 1, -1, 0, -1, -1, 0, 0, 1};
        }
    }

    std::vector<int32_t> HexHoodOddY(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 1, 0, 0, -1, -1, -1, -1, 0, -1, 1, 0, 1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0,
                                         1, 0, 0, -1, -1, -1, -1, 0, -1, 1, 0, 1};
        }
    }

    std::vector<int32_t> TriangleHoodSameParity(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0,
                                         0, 0, -1, 0, 1, 0, 0, 1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0,
                                         -1, 0, 1, 0, 0, 1};
        }
    }

    std::vector<int32_t> TriangleHoodDifParity(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0,
                                         0, 0, -1, 0, 1, 0, 0, -1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0,
                                         -1, 0, 1, 0, 0, -1};
        }
    }

    std::vector<int32_t> VonNeumannHood3D(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0,
                                         1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1};
        }
    }

    std::vector<int32_t> SemiMooreHood3D(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0,
                                         1, 0, 0,
                                         -1, 0, 0,
                                         0, 1, 0,
                                         0, -1, 0,
                                         0, 0, 1,
                                         0, 0, -1,
                                         1, 1, 0,
                                         1, -1, 0,
                                         1, 0, 1,
                                         1, 0, -1,
                                         -1, 1, 0,
                                         -1, -1, 0,
                                         -1, 0, 1,
                                         -1, 0, -1,
                                         0, 1, 1,
                                         0, 1, -1,
                                         0, -1, 1,
                                         0, -1, -1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         1, 0, 0,
                                         -1, 0, 0,
                                         0, 1, 0,
                                         0, -1, 0,
                                         0, 0, 1,
                                         0, 0, -1,
                                         1, 1, 0,
                                         1, -1, 0,
                                         1, 0, 1,
                                         1, 0, -1,
                                         -1, 1, 0,
                                         -1, -1, 0,
                                         -1, 0, 1,
                                         -1, 0, -1,
                                         0, 1, 1,
                                         0, 1, -1,
                                         0, -1, 1,
                                         0, -1, -1};
        }
    }

    std::vector<int32_t> MooreHood3D(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0,
                                         1, 0, 0,
                                         -1, 0, 0,
                                         0, 1, 0,
                                         0, -1, 0,
                                         0, 0, 1,
                                         0, 0, -1,
                                         1, 1, 0,
                                         1, -1, 0,
                                         1, 0, 1,
                                         1, 0, -1,
                                         -1, 1, 0,
                                         -1, -1, 0,
                                         -1, 0, 1,
                                         -1, 0, -1,
                                         0, 1, 1,
                                         0, 1, -1,
                                         0, -1, 1,
                                         0, -1, -1,
                                         1, 1, 1,
                                         1, 1, -1,
                                         1, -1, 1,
                                         1, -1, -1,
                                         -1, 1, 1,
                                         -1, 1, -1,
                                         -1, -1, 1,
                                         -1, -1, -1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         1, 0, 0,
                                         -1, 0, 0,
                                         0, 1, 0,
                                         0, -1, 0,
                                         0, 0, 1,
                                         0, 0, -1,
                                         1, 1, 0,
                                         1, -1, 0,
                                         1, 0, 1,
                                         1, 0, -1,
                                         -1, 1, 0,
                                         -1, -1, 0,
                                         -1, 0, 1,
                                         -1, 0, -1,
                                         0, 1, 1,
                                         0, 1, -1,
                                         0, -1, 1,
                                         0, -1, -1,
                                         1, 1, 1,
                                         1, 1, -1,
                                         1, -1, 1,
                                         1, -1, -1,
                                         -1, 1, 1,
                                         -1, 1, -1,
                                         -1, -1, 1,
                                         -1, -1, -1};
        }
    }

    std::vector<int32_t> CubicHoneyHood3DEvenZ(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0,
                                         0, 0, 2,
                                         0, 0, -2,
                                         1, 0, 0,
                                         -1, 0, 0,
                                         0, 1, 0,
                                         0, -1, 0,
                                         0, 0, 1,
                                         -1, 0, 1,
                                         -1, -1, 1,
                                         0, -1, 1,
                                         0, 0, -1,
                                         -1, 0, -1,
                                         -1, -1, -1,
                                         0, -1, -1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 2,
                                         0, 0, -2,
                                         1, 0, 0,
                                         -1, 0, 0,
                                         0, 1, 0,
                                         0, -1, 0,
                                         0, 0, 1,
                                         -1, 0, 1,
                                         -1, -1, 1,
                                         0, -1, 1,
                                         0, 0, -1,
                                         -1, 0, -1,
                                         -1, -1, -1,
                                         0, -1, -1};
        }
    }

    std::vector<int32_t> CubicHoneyHood3DOddZ(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0,
                                         0, 0, 2,
                                         0, 0, -2,
                                         1, 0, 0,
                                         -1, 0, 0,
                                         0, 1, 0,
                                         0, -1, 0,
                                         0, 0, 1,
                                         1, 0, 1,
                                         1, 1, 1,
                                         0, 1, 1,
                                         0, 0, -1,
                                         1, 0, -1,
                                         1, 1, -1,
                                         0, 1, -1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 2,
                                         0, 0, -2,
                                         1, 0, 0,
                                         -1, 0, 0,
                                         0, 1, 0,
                                         0, -1, 0,
                                         0, 0, 1,
                                         1, 0, 1,
                                         1, 1, 1,
                                         0, 1, 1,
                                         0, 0, -1,
                                         1, 0, -1,
                                         1, 1, -1,
                                         0, 1, -1};
        }
    }

    std::vector<int32_t> DiagonalHood3D(bool includeOrigin)
    {
        if (includeOrigin)
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0,
                                         1, 1, 1,
                                         1, 1, -1,
                                         1, -1, 1,
                                         1, -1, -1,
                                         -1, 1, 1,
                                         -1, 1, -1,
                                         -1, -1, 1,
                                         -1, -1, -1};
        }
        else
        {
            return std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0,
                                         1, 1, 1,
                                         1, 1, -1,
                                         1, -1, 1,
                                         1, -1, -1,
                                         -1, 1, 1,
                                         -1, 1, -1,
                                         -1, -1, 1,
                                         -1, -1, -1};
        }
    }
}
