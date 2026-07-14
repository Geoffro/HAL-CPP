#pragma once

namespace Hal
{
    class BinaryReader;
    class BinaryWriter;
    class Rand;

    // Marsaglia polar method Gaussian sampler, ported directly from HAL's
    // Tools/Internal/Gaussian.java — the algorithm has no Java-specific
    // dependency, so it carries over unchanged. Caches one of the two
    // values produced per iteration of the polar method, so state must
    // persist across calls (hence the member fields rather than a free
    // function).
    class GaussianSampler
    {
    public:
        GaussianSampler();
        ~GaussianSampler();

        double Sample(double  mean,
                      double  stdDev,
                      Rand   &rand);

        // Persists/restores the cached companion value (m_v1/m_v2/m_s) and
        // m_phase, so a reloaded sampler's next Sample() call reproduces
        // the exact same value the original would have — without this,
        // a mid-pair save/load would silently redraw both values of the
        // pair instead of returning the cached second one.
        void Save(BinaryWriter &writer) const;
        void Load(BinaryReader &reader);

    private:
        double m_v1;
        double m_v2;
        double m_s;
        double m_phase;
    };
}
