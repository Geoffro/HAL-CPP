#pragma once

#include <cstdint>

namespace Hal
{
    class BinaryReader;
    class BinaryWriter;


    // Core uniform pseudorandom generator. Implements xoshiro256** (Blackman
    // & Vigna, public domain), seeded via splitmix64, in place of porting
    // HAL's vendored java.util.SplittableRandom implementation verbatim (see
    // docs/DESIGN.md, "RNG"). The design goal HAL cared about — cheap,
    // statistically independent streams per instance — is preserved via
    // Split(), which reseeds a new generator from this one's own output
    // rather than sharing state.
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

        // Returns a new, independently-seeded Rng derived from this one's
        // output stream.
        Rng Split();

        // Persists/restores the full generator state (the 4-word
        // xoshiro256** state), so a reloaded Rng continues the exact same
        // output stream from where it left off.
        void Save(BinaryWriter &writer) const;
        void Load(BinaryReader &reader);

    private:
        uint64_t BoundedUint64(uint64_t bound);

        uint64_t m_state[4];
    };
}
