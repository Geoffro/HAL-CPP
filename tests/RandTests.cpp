#include <doctest/doctest.h>

#include "Hal/Rand.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

TEST_CASE("Double() produces values in [0,1) with mean and variance close to uniform")
{
    Hal::Rand rand(12345);
    const int32_t sampleCount = 200000;
    double sum = 0;
    double sumSquares = 0;
    for (int32_t i = 0; i < sampleCount; i++)
    {
        double v = rand.Double();
        REQUIRE(v >= 0.0);
        REQUIRE(v < 1.0);
        sum += v;
        sumSquares += v * v;
    }
    double mean = sum / sampleCount;
    double variance = sumSquares / sampleCount - mean * mean;

    // Uniform[0,1): mean = 0.5, variance = 1/12.
    CHECK(mean == doctest::Approx(0.5).epsilon(0.01));
    CHECK(variance == doctest::Approx(1.0 / 12.0).epsilon(0.02));
}

TEST_CASE("Int(bound) stays within [0,bound) and covers the full range")
{
    Hal::Rand rand(42);
    std::vector<bool> seen(10, false);
    for (int32_t i = 0; i < 10000; i++)
    {
        int32_t v = rand.Int(10);
        REQUIRE((v >= 0 && v < 10));
        seen[v] = true;
    }
    for (bool wasSeen : seen)
    {
        CHECK(wasSeen);
    }
}

TEST_CASE("Same seed produces the same sequence; different seeds diverge")
{
    Hal::Rand a(999);
    Hal::Rand b(999);
    Hal::Rand c(1000);

    for (int32_t i = 0; i < 20; i++)
    {
        CHECK(a.Double() == b.Double());
    }

    bool anyDifferent = false;
    for (int32_t i = 0; i < 20; i++)
    {
        if (a.Double() != c.Double())
        {
            anyDifferent = true;
        }
    }
    CHECK(anyDifferent);
}

TEST_CASE("Gaussian samples have the requested mean and standard deviation")
{
    Hal::Rand rand(7);
    const int32_t sampleCount = 200000;
    double sum = 0;
    double sumSquares = 0;
    for (int32_t i = 0; i < sampleCount; i++)
    {
        double v = rand.Gaussian(5.0, 2.0);
        sum += v;
        sumSquares += v * v;
    }
    double mean = sum / sampleCount;
    double variance = sumSquares / sampleCount - mean * mean;

    CHECK(mean == doctest::Approx(5.0).epsilon(0.01));
    CHECK(std::sqrt(variance) == doctest::Approx(2.0).epsilon(0.02));
}

TEST_CASE("Binomial sample mean converges to n*p across the fast-path thresholds")
{
    Hal::Rand rand(2024);

    // Exercises BinomialProbSample (n<10), rk_binomial_inversion (n*p<30),
    // and the BTPE/Colt path, matching SampleIntFast's branch conditions.
    struct Case
    {
        int32_t n;
        double p;
    };
    std::vector<Case> cases = {{5, 0.5}, {100, 0.1}, {1000, 0.3}};

    for (const Case &c : cases)
    {
        int64_t total = 0;
        const int32_t trials = 5000;
        for (int32_t i = 0; i < trials; i++)
        {
            total += rand.Binomial(c.n, c.p);
        }
        double mean = static_cast<double>(total) / trials;
        double expected = c.n * c.p;
        double tolerance = std::max(0.5, expected * 0.1);
        CHECK(mean == doctest::Approx(expected).epsilon(0.1).scale(tolerance));
    }
}

TEST_CASE("Shuffle produces a permutation of the original elements")
{
    Hal::Rand rand(55);
    std::vector<int32_t> arr(20);
    std::iota(arr.begin(), arr.end(), 0);

    rand.Shuffle(arr.data(), static_cast<int32_t>(arr.size()));

    std::vector<int32_t> sorted = arr;
    std::sort(sorted.begin(), sorted.end());
    for (int32_t i = 0; i < 20; i++)
    {
        CHECK(sorted[i] == i);
    }
}

TEST_CASE("RandomVariable samples the bin matching its cumulative probability")
{
    Hal::Rand rand(3);
    double probs[3] = {0.2, 0.3, 0.5};
    std::vector<int32_t> counts(3, 0);
    const int32_t trials = 100000;
    for (int32_t i = 0; i < trials; i++)
    {
        int32_t bin = rand.RandomVariable(probs, 3);
        REQUIRE((bin >= 0 && bin < 3));
        counts[bin]++;
    }
    CHECK(static_cast<double>(counts[0]) / trials == doctest::Approx(0.2).epsilon(0.05));
    CHECK(static_cast<double>(counts[1]) / trials == doctest::Approx(0.3).epsilon(0.05));
    CHECK(static_cast<double>(counts[2]) / trials == doctest::Approx(0.5).epsilon(0.05));
}
