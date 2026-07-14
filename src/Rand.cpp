#include "Hal/Rand.h"

#include "Hal/IO/BinaryReader.h"
#include "Hal/IO/BinaryWriter.h"
#include "Hal/MathUtil.h"

#include <cmath>

namespace Hal
{
    Rand::Rand(uint64_t seed)
        : m_rng(seed),
          m_gaussian(),
          m_binomial()
    {
    }

    Rand::Rand()
        : m_rng(),
          m_gaussian(),
          m_binomial()
    {
    }

    Rand::~Rand()
    {
    }

    int32_t Rand::Int(int32_t bound)
    {
        if (bound == 1)
        {
            return 0;
        }
        return m_rng.NextInt(bound);
    }

    double Rand::Double(double bound)
    {
        return m_rng.NextDouble(bound);
    }

    double Rand::Double(int32_t bound)
    {
        return m_rng.NextDouble(static_cast<double>(bound));
    }

    double Rand::Double()
    {
        return m_rng.NextDouble();
    }

    int64_t Rand::Long(int64_t bound)
    {
        return m_rng.NextLong(bound);
    }

    bool Rand::Bool()
    {
        return m_rng.NextBool();
    }

    int64_t Rand::Binomial(int64_t n,
                           double  p)
    {
        return m_binomial.SampleLongFast(n, p, *this);
    }

    int32_t Rand::Binomial(int32_t n,
                           double  p)
    {
        return m_binomial.SampleIntFast(n, p, *this);
    }

    double Rand::Gaussian(double mean,
                          double stdDev)
    {
        return m_gaussian.Sample(mean, stdDev, *this);
    }

    void Rand::Multinomial(const double *pProbabilities,
                           int32_t       length,
                           int32_t       n,
                           int32_t      *pRet)
    {
        double probRemaining = 1;
        for (int32_t i = 0; i < length; i++)
        {
            double prob = pProbabilities[i];
            if ((probRemaining - prob) <= 0)
            {
                pRet[i] = n;
                return;
            }
            int32_t popSelected = Binomial(n, prob / probRemaining);
            probRemaining -= prob;
            n -= popSelected;
            pRet[i] = popSelected;
        }
    }

    void Rand::Multinomial(const double *pProbabilities,
                           int32_t       length,
                           int64_t       n,
                           int64_t      *pRet)
    {
        double probRemaining = 1;
        for (int32_t i = 0; i < length; i++)
        {
            double prob = pProbabilities[i];
            if ((probRemaining - prob) <= 0)
            {
                pRet[i] = n;
                return;
            }
            int64_t popSelected = Binomial(n, prob / probRemaining);
            probRemaining -= prob;
            n -= popSelected;
            pRet[i] = popSelected;
        }
    }

    void Rand::RandomPointOnSphereEdge(double  radius,
                                       double *pRet)
    {
        double x = m_gaussian.Sample(0, radius, *this);
        double y = m_gaussian.Sample(0, radius, *this);
        double z = m_gaussian.Sample(0, radius, *this);
        double norm = Norm(x, y, z);
        pRet[0] = (x * radius) / norm;
        pRet[1] = (y * radius) / norm;
        pRet[2] = (z * radius) / norm;
    }

    void Rand::RandomPointInSphere(double  radius,
                                   double *pRet)
    {
        do
        {
            pRet[0] = Double();
            pRet[1] = Double();
            pRet[2] = Double();
        } while (DistSquared(pRet[0], pRet[1], pRet[2], 0.5, 0.5, 0.5) > 0.25);

        double retScale = radius * 2;
        pRet[0] = (pRet[0] - 0.5) * retScale;
        pRet[1] = (pRet[1] - 0.5) * retScale;
        pRet[2] = (pRet[2] - 0.5) * retScale;
    }

    void Rand::RandomPointOnCircleEdge(double  radius,
                                       double *pRet)
    {
        double x = m_gaussian.Sample(0, radius, *this);
        double y = m_gaussian.Sample(0, radius, *this);
        double norm = Norm(x, y);
        pRet[0] = (x * radius) / norm;
        pRet[1] = (y * radius) / norm;
    }

    void Rand::RandomPointInCircle(double  radius,
                                   double *pRet)
    {
        const double pi = 3.14159265358979323846;
        double r = std::sqrt(Double()) * radius;
        double a = Double() * pi * 2;
        pRet[0] = r * std::cos(a);
        pRet[1] = r * std::sin(a);
    }

    int32_t Rand::RandomVariable(const double *pProbs,
                                 int32_t       length)
    {
        double rand = Double();
        for (int32_t i = 0; i < length; i++)
        {
            rand -= pProbs[i];
            if (rand <= 0)
            {
                return i;
            }
        }
        return -1;
    }

    int32_t Rand::RandomVariable(const double *pProbs,
                                 int32_t       start,
                                 int32_t       end)
    {
        double rand = Double();
        for (int32_t i = start; i < end; i++)
        {
            rand -= pProbs[i];
            if (rand <= 0)
            {
                return i;
            }
        }
        return -1;
    }

    int32_t Rand::RandomVariable(const double *pProbs,
                                 double        rand,
                                 int32_t       start,
                                 int32_t       end)
    {
        for (int32_t i = start; i < end; i++)
        {
            rand -= pProbs[i];
            if (rand <= 0)
            {
                return i;
            }
        }
        return -1;
    }

    void Rand::RandomDS(double  *pOut,
                        int32_t  length,
                        double   min,
                        double   max)
    {
        for (int32_t i = 0; i < length; i++)
        {
            pOut[i] = Double() * (max - min) + min;
        }
    }

    void Rand::RandomIS(int32_t *pOut,
                        int32_t  length,
                        int32_t  min,
                        int32_t  max)
    {
        for (int32_t i = 0; i < length; i++)
        {
            pOut[i] = Int(max - min) + min;
        }
    }

    double Rand::ExponentialDist(double rate)
    {
        return -std::log(Double()) / rate;
    }

    double Rand::WeibullDist(double shape,
                             double scale)
    {
        return scale * std::pow(-std::log(1 - Double()), (1 / shape));
    }

    void Rand::Save(BinaryWriter &writer) const
    {
        m_rng.Save(writer);
        m_gaussian.Save(writer);
    }

    void Rand::Load(BinaryReader &reader)
    {
        m_rng.Load(reader);
        m_gaussian.Load(reader);
    }
}
