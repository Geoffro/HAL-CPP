#include <doctest/doctest.h>

#include "Hal/Rand.h"
#include "Hal/Tools/MultinomialCalc.h"

#include <cstdint>
#include <vector>

// The conditional-binomial decomposition must conserve the population
// exactly and reproduce pop*prob per outcome on average.

TEST_CASE("MultinomialCalc conserves the population across a full set of outcomes")
{
    Hal::Rand            rng(12345);
    Hal::MultinomialCalc mn(rng);

    std::vector<double> probs = {0.1, 0.2, 0.3, 0.4};
    int32_t             pop = 1000;

    mn.Setup(pop);
    int32_t total = 0;
    for (double p : probs)
    {
        total += mn.Sample(p);
    }
    // Every individual is assigned to exactly one outcome.
    CHECK(total == pop);
    CHECK(mn.GetPopRemaining() == 0);
}

TEST_CASE("MultinomialCalc: a zero-probability outcome takes nobody; a certain one takes everyone")
{
    Hal::Rand            rng(7);
    Hal::MultinomialCalc mn(rng);

    mn.Setup(500);
    CHECK(mn.Sample(0.0) == 0);
    CHECK(mn.GetPopRemaining() == 500);
    // Probability consuming all the remaining mass scoops up the rest.
    CHECK(mn.Sample(1.0) == 500);
    CHECK(mn.GetPopRemaining() == 0);
}

TEST_CASE("MultinomialCalc per-outcome mean tracks pop*prob over many trials")
{
    Hal::Rand            rng(999);
    Hal::MultinomialCalc mn(rng);

    std::vector<double> probs = {0.25, 0.25, 0.5};
    int32_t             pop = 800;
    int32_t             trials = 4000;
    std::vector<double> sums(probs.size(), 0.0);

    for (int32_t t = 0; t < trials; t++)
    {
        mn.Setup(pop);
        for (size_t k = 0; k < probs.size(); k++)
        {
            sums[k] += mn.Sample(probs[k]);
        }
    }

    for (size_t k = 0; k < probs.size(); k++)
    {
        double mean = sums[k] / trials;
        double expected = pop * probs[k];
        // Within ~3% of the expected count.
        CHECK(mean == doctest::Approx(expected).epsilon(0.03));
    }
}

TEST_CASE("MultinomialCalc handles populations beyond the int32 range")
{
    Hal::Rand            rng(3);
    Hal::MultinomialCalc mn(rng);
    const int64_t pop = 5000000000LL;
    mn.Setup(pop);
    int64_t a = mn.Sample(0.25);
    int64_t b = mn.Sample(0.75);
    CHECK((a + b) == pop);
    CHECK(mn.GetPopRemaining() == 0);
    CHECK(static_cast<double>(a) == doctest::Approx(pop * 0.25).epsilon(0.001));
}
