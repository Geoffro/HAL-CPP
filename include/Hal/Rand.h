#pragma once

#include "Hal/BinomialSampler.h"
#include "Hal/GaussianSampler.h"
#include "Hal/Rng.h"

#include <cstdint>

namespace Hal
{
    class BinaryReader;
    class BinaryWriter;

    // Random value generation and distribution sampling: a core generator
    // plus Gaussian and binomial samplers.
    class Rand
    {
    public:
        explicit Rand(uint64_t seed);
        Rand();
        ~Rand();

        // Returns a random integer in [0, bound).
        int32_t Int(int32_t bound);

        // Returns a random double in [0, bound).
        double Double(double bound);
        double Double(int32_t bound);

        // Returns a random double in [0, 1).
        double Double();

        // Returns a random integer in [0, bound).
        int64_t Long(int64_t bound);

        bool Bool();

        // Number of heads from n weighted coin flips with probability p of
        // heads. Throws std::invalid_argument if p is outside [0, 1].
        int64_t Binomial(int64_t n,
                         double  p);

        int32_t Binomial(int32_t n,
                         double  p);

        double Gaussian(double mean,
                        double stdDev);

        // Evaluates the multinomial on the input list of pProbabilities and
        // puts the resulting number of individuals from n into pRet. The sum
        // of pProbabilities must be <= 1.
        void Multinomial(const double *pProbabilities,
                         int32_t       length,
                         int32_t       n,
                         int32_t      *pRet);

        void Multinomial(const double *pProbabilities,
                         int32_t       length,
                         int64_t       n,
                         int64_t      *pRet);

        // Writes a random point on the surface of a sphere of the given
        // radius, centered at the origin, into pRet[0..2].
        void RandomPointOnSphereEdge(double  radius,
                                     double *pRet);

        void RandomPointInSphere(double  radius,
                                 double *pRet);

        // Writes a random point on the edge of a circle of the given
        // radius, centered at the origin, into pRet[0..1].
        void RandomPointOnCircleEdge(double  radius,
                                     double *pRet);

        void RandomPointInCircle(double  radius,
                                 double *pRet);

        // Samples a discrete random variable from the pProbabilities
        // provided (should sum to 1). Returns the sampled index, or -1.
        int32_t RandomVariable(const double *pProbs,
                               int32_t       length);

        int32_t RandomVariable(const double *pProbs,
                               int32_t       start,
                               int32_t       end);

        int32_t RandomVariable(const double *pProbs,
                               double        rand,
                               int32_t       start,
                               int32_t       end);

        // Fills pOut[0..length) with random doubles in [min, max].
        void RandomDS(double  *pOut,
                      int32_t  length,
                      double   min,
                      double   max);

        // Fills pOut[0..length) with random integers in [min, max).
        void RandomIS(int32_t *pOut,
                      int32_t  length,
                      int32_t  min,
                      int32_t  max);

        // Samples an exponential distribution with the given rate parameter
        // (equivalent to the timing of the next Poisson event).
        double ExponentialDist(double rate);

        // Samples a Weibull distribution via its inverse cumulative function.
        double WeibullDist(double shape,
                           double scale);

        // Fisher-Yates shuffle. sampleSize elements are eligible to be
        // drawn from; numberOfShuffles elements at the front are shuffled.
        template <typename T>
        void Shuffle(T       *pArr,
                     int32_t  sampleSize,
                     int32_t  numberOfShuffles)
        {
            for (int32_t i = 0; i < numberOfShuffles; i++)
            {
                int32_t iSwap = Int(sampleSize - i) + i;
                T swap = pArr[iSwap];
                pArr[iSwap] = pArr[i];
                pArr[i] = swap;
            }
        }

        template <typename T>
        void Shuffle(T       *pArr,
                     int32_t  length)
        {
            Shuffle(pArr, length, length);
        }

        // Persists/restores the generator and Gaussian state, so a reloaded
        // Rand reproduces the same future output. BinomialSampler's cache
        // only memoizes per-(n, p) constants and isn't saved.
        void Save(BinaryWriter &writer) const;
        void Load(BinaryReader &reader);

    private:
        Rng m_rng;
        GaussianSampler m_gaussian;
        BinomialSampler m_binomial;
    };
}
