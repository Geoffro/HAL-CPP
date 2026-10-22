// Headless: a non-spatial model of a mutating population tracked as
// per-mutation-class counts. Each step every class dies, divides
// (density-limited), and passes a fraction of its divisions to the next
// mutation class, all as binomial draws. Prints the population vector
// periodically.

#include "Hal/Rand.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

class NonSpatialAbm
{
public:
    explicit NonSpatialAbm(std::vector<int64_t> pops)
        : m_pops(std::move(pops))
    {
        m_totalPop = Sum();
    }

    double DivProb(int32_t iPop) const
    {
        return 0.5 * std::pow(0.9, iPop) * (1.0 - m_totalPop * 1.0 / 10000.0);
    }
    double DieProb(int32_t) const { return 0.01; }
    double MutProb(int32_t) const { return 0.1; }

    void Step(Hal::Rand &rng)
    {
        int32_t last = static_cast<int32_t>(m_pops.size()) - 1;
        for (int32_t i = last; i >= 0; i--)
        {
            m_pops[i] -= rng.Binomial(m_pops[i], DieProb(i));
            int64_t nDivs = rng.Binomial(m_pops[i], DivProb(i));
            m_pops[i] += nDivs;
            if (i != last)
            {
                int64_t nMuts = rng.Binomial(nDivs, MutProb(i));
                m_pops[i] -= nMuts;
                m_pops[i + 1] += nMuts;
            }
        }
        m_totalPop = Sum();
    }

    const std::vector<int64_t> &Pops() const { return m_pops; }
    int64_t                     TotalPop() const { return m_totalPop; }

private:
    int64_t Sum() const
    {
        int64_t s = 0;
        for (int64_t v : m_pops)
        {
            s += v;
        }
        return s;
    }

    std::vector<int64_t> m_pops;
    int64_t              m_totalPop;
};

int main()
{
    Hal::Rand            rng;
    std::vector<int64_t> startPops(10, 0);
    startPops[0] = 100;
    NonSpatialAbm model(startPops);

    // Print every 1000th step to keep the output readable.
    for (int32_t i = 0; i < 100000; i++)
    {
        model.Step(rng);
        if ((i % 1000) == 0)
        {
            for (int64_t v : model.Pops())
            {
                std::printf("%lld,", static_cast<long long>(v));
            }
            std::printf("%lld\n", static_cast<long long>(model.TotalPop()));
        }
    }
    return 0;
}
