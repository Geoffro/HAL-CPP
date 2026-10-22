// Headless: draws one binomial sample with a huge n and confirms the
// sample fraction is near p.

#include "Hal/Rand.h"

#include <cstdint>
#include <cstdio>

int main()
{
    Hal::Rand rng;
    int64_t   n = INT64_MAX;
    double    fraction = rng.Binomial(n, 0.1) * 1.0 / static_cast<double>(n);
    std::printf("%g\n", fraction);
    return 0;
}
