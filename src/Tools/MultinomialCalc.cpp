#include "Hal/Tools/MultinomialCalc.h"

namespace Hal
{
    MultinomialCalc::MultinomialCalc(Rand &rng)
        : m_rng(rng),
          m_popRemaining(0),
          m_probRemaining(1.0)
    {
    }

    void MultinomialCalc::Setup(int32_t pop)
    {
        m_popRemaining = pop;
        m_probRemaining = 1.0;
    }

    int32_t MultinomialCalc::Sample(double prob)
    {
        if ((m_popRemaining == 0) || (prob == 0.0))
        {
            return 0;
        }
        if ((m_probRemaining - prob) <= 0.0)
        {
            int32_t ret = m_popRemaining;
            m_popRemaining = 0;
            m_probRemaining -= prob;
            return ret;
        }
        int32_t popSelected = m_rng.Binomial(m_popRemaining, prob / m_probRemaining);
        m_probRemaining -= prob;
        m_popRemaining -= popSelected;
        return popSelected;
    }
}
