#include <doctest/doctest.h>

#include "Hal/MathUtil.h"
#include "Hal/Rand.h"
#include "Hal/Rng.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
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

TEST_CASE("Binomial int32 and int64 draws with the same p stay within [0, n] when interleaved")
{
    Hal::Rand rand(7);
    int64_t smallTotal = 0;
    const int32_t trials = 2000;
    for (int32_t i = 0; i < trials; i++)
    {
        int32_t small = rand.Binomial(static_cast<int32_t>(1000), 0.3);
        int64_t large = rand.Binomial(static_cast<int64_t>(100000), 0.3);
        REQUIRE(small >= 0);
        REQUIRE(small <= 1000);
        REQUIRE(large >= 0);
        REQUIRE(large <= 100000);
        smallTotal += small;
    }
    CHECK((static_cast<double>(smallTotal) / trials) == doctest::Approx(300.0).epsilon(0.02));
}

TEST_CASE("Long and Bool cover their ranges evenly")
{
    Hal::Rand rand(11);
    const int64_t bound = 5000000000LL;
    int64_t maxSeen = 0;
    int32_t trueCount = 0;
    const int32_t trials = 20000;
    for (int32_t i = 0; i < trials; i++)
    {
        int64_t v = rand.Long(bound);
        REQUIRE(v >= 0);
        REQUIRE(v < bound);
        maxSeen = std::max(maxSeen, v);
        if (rand.Bool())
        {
            trueCount++;
        }
    }
    CHECK(maxSeen > (bound / 2));
    CHECK((static_cast<double>(trueCount) / trials) == doctest::Approx(0.5).epsilon(0.05));
}

TEST_CASE("Rng rejects non-positive bounds and Split derives an independent stream")
{
    Hal::Rng rng(5);
    CHECK_THROWS_AS(rng.NextInt(0), std::invalid_argument);
    CHECK_THROWS_AS(rng.NextLong(-1), std::invalid_argument);

    Hal::Rng child = rng.Split();
    int32_t same = 0;
    for (int32_t i = 0; i < 100; i++)
    {
        if (child.NextUint64() == rng.NextUint64())
        {
            same++;
        }
    }
    CHECK(same == 0);
}

static void CheckBinomialMoments(Hal::Rand &rand,
                                 int64_t    n,
                                 double     p,
                                 bool       useInt32)
{
    const int32_t trials = 20000;
    double sum = 0;
    double sumSquares = 0;
    for (int32_t i = 0; i < trials; i++)
    {
        double v = useInt32 ? static_cast<double>(rand.Binomial(static_cast<int32_t>(n), p)) : static_cast<double>(rand.Binomial(n, p));
        REQUIRE(v >= 0);
        REQUIRE(v <= static_cast<double>(n));
        sum += v;
        sumSquares += v * v;
    }
    double mean = sum / trials;
    double variance = sumSquares / trials - mean * mean;
    double expectedVariance = static_cast<double>(n) * p * (1 - p);
    CAPTURE(n);
    CAPTURE(p);
    CAPTURE(useInt32);
    CHECK(mean == doctest::Approx(static_cast<double>(n) * p).epsilon(0.01));
    CHECK(variance == doctest::Approx(expectedVariance).epsilon(0.05));
}

TEST_CASE("Binomial matches the expected mean and variance on every sampling path, for int32 and int64")
{
    Hal::Rand rand(99);
    for (int32_t useInt32 = 0; useInt32 < 2; useInt32++)
    {
        CheckBinomialMoments(rand, 200, 0.05, useInt32 == 1);
        CheckBinomialMoments(rand, 200, 0.97, useInt32 == 1);
        CheckBinomialMoments(rand, 1000, 0.3, useInt32 == 1);
        CheckBinomialMoments(rand, 1000, 0.8, useInt32 == 1);
        CheckBinomialMoments(rand, 100000, 0.5, useInt32 == 1);
    }
}

TEST_CASE("Binomial handles populations beyond the int32 range")
{
    Hal::Rand rand(17);
    const int64_t n = 6000000000LL;
    const double p = 0.5;
    double sigma = std::sqrt(static_cast<double>(n) * p * (1 - p));
    for (int32_t i = 0; i < 200; i++)
    {
        int64_t v = rand.Binomial(n, p);
        CHECK(std::abs(static_cast<double>(v) - static_cast<double>(n) * p) < (6 * sigma));
    }
}

TEST_CASE("Binomial rejects probabilities outside [0, 1] for both count types")
{
    Hal::Rand rand(1);
    CHECK_THROWS_AS(rand.Binomial(static_cast<int32_t>(10), 1.5), std::invalid_argument);
    CHECK_THROWS_AS(rand.Binomial(static_cast<int32_t>(10), -0.1), std::invalid_argument);
    CHECK_THROWS_AS(rand.Binomial(static_cast<int64_t>(10), 1.5), std::invalid_argument);
    CHECK(rand.Binomial(static_cast<int32_t>(10), 1.0) == 10);
    CHECK(rand.Binomial(static_cast<int64_t>(10), 0.0) == 0);
}

TEST_CASE("Multinomial assigns the whole population across the outcomes")
{
    Hal::Rand rand(4);
    const double probs[4] = {0.1, 0.2, 0.3, 0.4};
    int64_t sums[4] = {0, 0, 0, 0};
    const int32_t trials = 2000;
    for (int32_t trial = 0; trial < trials; trial++)
    {
        int32_t counts32[4] = {0, 0, 0, 0};
        rand.Multinomial(probs, 4, static_cast<int32_t>(1000), counts32);
        CHECK((counts32[0] + counts32[1] + counts32[2] + counts32[3]) == 1000);

        int64_t counts64[4] = {0, 0, 0, 0};
        rand.Multinomial(probs, 4, static_cast<int64_t>(1000), counts64);
        CHECK((counts64[0] + counts64[1] + counts64[2] + counts64[3]) == 1000);
        for (int32_t k = 0; k < 4; k++)
        {
            sums[k] += counts64[k];
        }
    }
    for (int32_t k = 0; k < 4; k++)
    {
        CAPTURE(k);
        CHECK((static_cast<double>(sums[k]) / trials) == doctest::Approx(1000 * probs[k]).epsilon(0.02));
    }
}

TEST_CASE("Random points land on or within circles and spheres of the given radius")
{
    Hal::Rand rand(8);
    for (int32_t i = 0; i < 1000; i++)
    {
        double point[3] = {0, 0, 0};
        rand.RandomPointOnCircleEdge(2.0, point);
        CHECK(Hal::Norm(point[0], point[1]) == doctest::Approx(2.0));
        rand.RandomPointInCircle(2.0, point);
        CHECK(Hal::Norm(point[0], point[1]) <= 2.0);
        rand.RandomPointOnSphereEdge(3.0, point);
        CHECK(Hal::Norm(point[0], point[1], point[2]) == doctest::Approx(3.0));
        rand.RandomPointInSphere(3.0, point);
        CHECK(Hal::Norm(point[0], point[1], point[2]) <= 3.0);
    }
}

TEST_CASE("RandomDS and RandomIS fill arrays within their ranges")
{
    Hal::Rand rand(2);
    std::vector<double> doubles(500);
    std::vector<int32_t> ints(500);
    rand.RandomDS(doubles.data(), 500, -2.0, 3.0);
    rand.RandomIS(ints.data(), 500, -4, 4);
    for (int32_t i = 0; i < 500; i++)
    {
        CHECK(doubles[static_cast<size_t>(i)] >= -2.0);
        CHECK(doubles[static_cast<size_t>(i)] < 3.0);
        CHECK(ints[static_cast<size_t>(i)] >= -4);
        CHECK(ints[static_cast<size_t>(i)] < 4);
    }
}

TEST_CASE("ExponentialDist and WeibullDist have the expected means")
{
    Hal::Rand rand(21);
    const int32_t trials = 100000;
    double expSum = 0;
    double weibullSum = 0;
    for (int32_t i = 0; i < trials; i++)
    {
        expSum += rand.ExponentialDist(4.0);
        weibullSum += rand.WeibullDist(1.0, 3.0);
    }
    CHECK((expSum / trials) == doctest::Approx(0.25).epsilon(0.02));
    CHECK((weibullSum / trials) == doctest::Approx(3.0).epsilon(0.02));
}
