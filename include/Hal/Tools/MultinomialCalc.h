#pragma once

#include "Hal/Rand.h"

#include <cstdint>

namespace Hal
{
    // Draws a multinomial one outcome at a time. Call Setup(pop), then
    // Sample(prob) for each outcome; each call returns how many of the
    // not-yet-assigned population fall into that outcome (a conditional
    // binomial draw from the shared Rand). The probabilities should sum to
    // at most 1; an outcome that exhausts the remaining probability takes
    // the whole remaining population.
    class MultinomialCalc
    {
    public:
        explicit MultinomialCalc(Rand &rng);

        void Setup(int64_t pop);

        // Count assigned to an outcome with the given (unconditional)
        // probability.
        int64_t Sample(double prob);

        int64_t GetPopRemaining() const;
        double GetProbRemaining() const;

    private:
        Rand   &m_rng;
        int64_t m_popRemaining;
        double  m_probRemaining;
    };
}
