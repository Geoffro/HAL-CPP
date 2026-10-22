// Headless: fills a PDEGrid2D with a vertical ramp, then prints the
// finite-difference x/y gradient at an interior cell.

#include "Hal/Grids/PDEGrid2D.h"

#include <cstdio>

int main()
{
    const int32_t xDim = 10;
    const int32_t yDim = 10;
    Hal::PDEGrid2D diff(xDim, yDim);

    for (int32_t y = 0; y < yDim; y++)
    {
        double setVal = y * 1.0 / xDim;
        for (int32_t x = 0; x < xDim; x++)
        {
            diff.Set(x, y, setVal);
        }
    }
    diff.Update();

    std::printf("y gradient: %g\n", diff.GradientY(5, 5));
    std::printf("x gradient: %g\n", diff.GradientX(5, 5));
    return 0;
}
