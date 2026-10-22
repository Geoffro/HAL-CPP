// Headless: fills a PDEGrid3D with a ramp along z, then prints the
// finite-difference gradient at an interior cell along each axis.

#include "Hal/Grids/PDEGrid3D.h"

#include <cstdio>

int main()
{
    const int32_t xDim = 10;
    const int32_t yDim = 10;
    const int32_t zDim = 10;
    Hal::PDEGrid3D diff(xDim, yDim, zDim);

    for (int32_t z = 0; z < zDim; z++)
    {
        double setVal = z * 1.0 / xDim;
        for (int32_t y = 0; y < yDim; y++)
        {
            for (int32_t x = 0; x < xDim; x++)
            {
                diff.Set(x, y, z, setVal);
            }
        }
    }
    diff.Update();

    std::printf("x gradient: %g\n", diff.GradientX(5, 5, 5));
    std::printf("y gradient: %g\n", diff.GradientY(5, 5, 5));
    std::printf("z gradient: %g\n", diff.GradientZ(5, 5, 5));
    return 0;
}
