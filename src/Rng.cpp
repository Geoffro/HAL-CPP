#include "Hal/Rng.h"

#include "Hal/IO/BinaryReader.h"
#include "Hal/IO/BinaryWriter.h"

#include <random>
#include <stdexcept>

namespace Hal
{
    namespace Internal
    {
        uint64_t SplitMix64Next(uint64_t &state)
        {
            state += 0x9E3779B97F4A7C15ULL;
            uint64_t z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        }

        uint64_t RotateLeft(uint64_t x,
                            int32_t  k)
        {
            return (x << k) | (x >> (64 - k));
        }

        uint64_t SeedFromEntropy()
        {
            std::random_device device;
            uint64_t high = static_cast<uint64_t>(device());
            uint64_t low = static_cast<uint64_t>(device());
            return (high << 32) ^ low;
        }
    }

    Rng::Rng(uint64_t seed)
        : m_state{0, 0, 0, 0}
    {
        uint64_t splitState = seed;
        m_state[0] = Internal::SplitMix64Next(splitState);
        m_state[1] = Internal::SplitMix64Next(splitState);
        m_state[2] = Internal::SplitMix64Next(splitState);
        m_state[3] = Internal::SplitMix64Next(splitState);
    }

    Rng::Rng()
        : Rng(Internal::SeedFromEntropy())
    {
    }

    Rng::~Rng()
    {
    }

    uint64_t Rng::NextUint64()
    {
        const uint64_t result = Internal::RotateLeft(m_state[1] * 5, 7) * 9;
        const uint64_t t = m_state[1] << 17;

        m_state[2] ^= m_state[0];
        m_state[3] ^= m_state[1];
        m_state[1] ^= m_state[2];
        m_state[0] ^= m_state[3];
        m_state[2] ^= t;
        m_state[3] = Internal::RotateLeft(m_state[3], 45);

        return result;
    }

    uint64_t Rng::BoundedUint64(uint64_t bound)
    {
        const uint64_t threshold = (~bound + 1) % bound;
        uint64_t r = NextUint64();
        while (r < threshold)
        {
            r = NextUint64();
        }
        return r % bound;
    }

    int32_t Rng::NextInt(int32_t bound)
    {
        if (bound <= 0)
        {
            throw std::invalid_argument("Rng::NextInt bound must be positive");
        }
        return static_cast<int32_t>(BoundedUint64(static_cast<uint64_t>(bound)));
    }

    double Rng::NextDouble()
    {
        // Top 53 bits give a uniform double in [0, 1).
        const uint64_t mantissaBits = NextUint64() >> 11;
        return static_cast<double>(mantissaBits) * (1.0 / 9007199254740992.0);
    }

    double Rng::NextDouble(double bound)
    {
        return NextDouble() * bound;
    }

    int64_t Rng::NextLong(int64_t bound)
    {
        if (bound <= 0)
        {
            throw std::invalid_argument("Rng::NextLong bound must be positive");
        }
        return static_cast<int64_t>(BoundedUint64(static_cast<uint64_t>(bound)));
    }

    bool Rng::NextBool()
    {
        return (NextUint64() >> 63) != 0;
    }

    Rng Rng::Split()
    {
        return Rng(NextUint64());
    }

    void Rng::Save(BinaryWriter &writer) const
    {
        writer.WriteArray(m_state, 4);
    }

    void Rng::Load(BinaryReader &reader)
    {
        reader.ReadArray(m_state, 4);
    }
}
