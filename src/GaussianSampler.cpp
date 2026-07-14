#include "Hal/GaussianSampler.h"

#include "Hal/IO/BinaryReader.h"
#include "Hal/IO/BinaryWriter.h"
#include "Hal/Rand.h"

#include <cmath>

namespace Hal
{
    GaussianSampler::GaussianSampler()
        : m_v1(0),
          m_v2(0),
          m_s(0),
          m_phase(0)
    {
    }

    GaussianSampler::~GaussianSampler()
    {
    }

    double GaussianSampler::Sample(double  mean,
                                   double  stdDev,
                                   Rand   &rand)
    {
        double x;
        if (m_phase == 0)
        {
            do
            {
                double u1 = rand.Double();
                double u2 = rand.Double();

                m_v1 = 2 * u1 - 1;
                m_v2 = 2 * u2 - 1;
                m_s = m_v1 * m_v1 + m_v2 * m_v2;
            } while ((m_s >= 1) || (m_s == 0));

            x = m_v1 * std::sqrt(-2 * std::log(m_s) / m_s);
        }
        else
        {
            x = m_v2 * std::sqrt(-2 * std::log(m_s) / m_s);
        }
        m_phase = 1 - m_phase;
        return x * stdDev + mean;
    }

    void GaussianSampler::Save(BinaryWriter &writer) const
    {
        writer.WriteDouble(m_v1);
        writer.WriteDouble(m_v2);
        writer.WriteDouble(m_s);
        writer.WriteDouble(m_phase);
    }

    void GaussianSampler::Load(BinaryReader &reader)
    {
        m_v1 = reader.ReadDouble();
        m_v2 = reader.ReadDouble();
        m_s = reader.ReadDouble();
        m_phase = reader.ReadDouble();
    }
}
