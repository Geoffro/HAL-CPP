#pragma once

#include <cstdint>

namespace Hal
{
    class BinaryReader;
    class BinaryWriter;

    // Uniform pseudorandom generator: xoshiro256** (Blackman & Vigna),
    // seeded via splitmix64. Split() derives an independent generator from
    // this one's output.
    class Rng
    {
    public:
        explicit Rng(uint64_t seed);
        Rng();
        ~Rng();

        uint64_t NextUint64();

        // Uniform integer in [0, bound).
        int32_t NextInt(int32_t bound);

        // Uniform double in [0, 1).
        double NextDouble();

        // Uniform double in [0, bound).
        double NextDouble(double bound);

        // Uniform integer in [0, bound).
        int64_t NextLong(int64_t bound);

        bool NextBool();

        Rng Split();

        // Persists/restores the full generator state, so a reloaded Rng
        // continues the same output stream.
        void Save(BinaryWriter &writer) const;
        void Load(BinaryReader &reader);

    private:
        uint64_t BoundedUint64(uint64_t bound);

        uint64_t m_state[4];
    };
}
