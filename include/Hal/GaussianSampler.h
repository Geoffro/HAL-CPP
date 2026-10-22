#pragma once

namespace Hal
{
    class BinaryReader;
    class BinaryWriter;
    class Rand;

    // Marsaglia polar method Gaussian sampler. Each iteration produces two
    // values; the second is cached and returned by the next call.
    class GaussianSampler
    {
    public:
        GaussianSampler();
        ~GaussianSampler();

        double Sample(double  mean,
                      double  stdDev,
                      Rand   &rand);

        // Persists/restores the cached second value, so a reloaded sampler
        // continues the same sequence.
        void Save(BinaryWriter &writer) const;
        void Load(BinaryReader &reader);

    private:
        double m_v1;
        double m_v2;
        double m_s;
        double m_phase;
    };
}
