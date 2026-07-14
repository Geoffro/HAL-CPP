#pragma once

#include "Hal/Rand.h"

#include <cstdint>

namespace Hal
{
    // Draws successive counts from a multinomial distribution one outcome at
    // a time, ported from Tools/MultinomialCalc.java. Call Setup(pop) with
    // the total population, then Sample(prob) repeatedly with each outcome's
    // probability; each call returns how many of the still-unassigned
    // population fall into that outcome, using a conditional-binomial
    // decomposition so the running totals stay exact. The probabilities are
    // expected to sum to 1 across the Sample calls following a Setup; the
    // final outcome scoops up whatever population remains.
    //
    // Unlike Java (which extends Rand), this holds a reference to an existing
    // Rand and draws from its stream, so the multinomial draws advance the
    // same generator the caller passed in.
    class MultinomialCalc
    {
    public:
        explicit MultinomialCalc(Rand &rng);

        void Setup(int32_t pop);

        // Count assigned to an outcome with the given (unconditional)
        // probability, drawn from the population not yet assigned.
        int32_t Sample(double prob);

        int32_t GetPopRemaining() const { return m_popRemaining; }
        double  GetProbRemaining() const { return m_probRemaining; }

    private:
        Rand   &m_rng;
        int32_t m_popRemaining;
        double  m_probRemaining;
    };
}
