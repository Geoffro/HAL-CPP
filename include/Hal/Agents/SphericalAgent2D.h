#pragma once

#include "Hal/Agents/AgentPT2D.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/MathUtil.h"
#include "Hal/Rand.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace Hal
{
    // Circular off-lattice agent that repels or attracts other agents on
    // its grid through a caller-supplied overlap-force law.
    template <typename TGrid>
    class SphericalAgent2D : public AgentPT2D<TGrid>
    {
    public:
        explicit SphericalAgent2D(TGrid *pGrid)
            : AgentPT2D<TGrid>(pGrid),
              m_radius(0),
              m_xVel(0),
              m_yVel(0)
        {
        }

        ~SphericalAgent2D()
        {
        }

        double Radius() const
        {
            return m_radius;
        }

        void SetRadius(double radius)
        {
            m_radius = radius;
        }

        double XVel() const
        {
            return m_xVel;
        }

        double YVel() const
        {
            return m_yVel;
        }

        void SetVelocity(double xVel,
                         double yVel)
        {
            m_xVel = xVel;
            m_yVel = yVel;
        }

        void DefaultInit(double radius)
        {
            m_radius = radius;
            m_xVel = 0;
            m_yVel = 0;
        }

        // interactionRad is the max distance apart two interacting agent
        // centers can be. overlapFun(touchDist, otherAgent) -> force:
        // touchDist > 0 means the two agents overlap by that distance,
        // < 0 means they're separated by that distance. A positive force
        // pushes this agent away from the other; negative pulls it closer.
        // Adds to xVel/yVel for every other agent within interactionRad
        // (agents at exactly the same position are skipped, having no
        // direction) and returns the sum of positive (repulsive) forces.
        template <typename OverlapFn>
        double SumForces(double    interactionRad,
                         OverlapFn overlapFun)
        {
            std::vector<typename TGrid::AgentType *> candidates;
            this->Grid()->GetAgentsRadApprox(candidates, this->Xpt(), this->Ypt(), interactionRad);
            double sum = 0;
            for (typename TGrid::AgentType *pOther : candidates)
            {
                if (pOther != this)
                {
                    double xComp = this->Grid()->DispX(this->Xpt(), pOther->Xpt());
                    double yComp = this->Grid()->DispY(this->Ypt(), pOther->Ypt());
                    double dist = Norm(xComp, yComp);
                    if ((dist < interactionRad) && (dist > 0.0))
                    {
                        double touchDist = (m_radius + pOther->Radius()) - dist;
                        double force = overlapFun(touchDist, *pOther);
                        m_xVel -= (xComp / dist) * force;
                        m_yVel -= (yComp / dist) * force;
                        if (force > 0)
                        {
                            sum += std::abs(force);
                        }
                    }
                }
            }
            return sum;
        }

        // Same force law, over a neighbor list and displacement info from
        // AgentGrid2D::GetAgentsRad queried at this agent's position.
        template <typename TNeighbor, typename OverlapFn>
        double SumForces(const std::vector<TNeighbor *>             &neighbors,
                         const std::vector<AgentDisplacementInfo2D> &displacementInfo,
                         OverlapFn                                   overlapFun)
        {
            double sum = 0;
            for (size_t i = 0; i < neighbors.size(); i++)
            {
                TNeighbor                     *pOther = neighbors[i];
                const AgentDisplacementInfo2D &info = displacementInfo[i];
                if ((pOther != this) && (info.dist > 0.0))
                {
                    double touchDist = (m_radius + pOther->Radius()) - info.dist;
                    double force = overlapFun(touchDist, *pOther);
                    m_xVel -= (info.dispX / info.dist) * force;
                    m_yVel -= (info.dispY / info.dist) * force;
                    if (force > 0)
                    {
                        sum += std::abs(force);
                    }
                }
            }
            return sum;
        }

        // Multiplies xVel/yVel by frictionConst: 1 applies no friction, 0
        // stops the agent immediately.
        void ApplyFriction(double frictionConst)
        {
            m_xVel *= frictionConst;
            m_yVel *= frictionConst;
        }

        // Caps the (xVel,yVel) vector's norm to maxVel.
        void CapVelocity(double maxVel)
        {
            double maxVelSq = maxVel * maxVel;
            double normSq = NormSquared(m_xVel, m_yVel);
            if (normSq > maxVelSq)
            {
                double convFactor = maxVel / std::sqrt(normSq);
                m_xVel *= convFactor;
                m_yVel *= convFactor;
            }
        }

        // Applies the accumulated xVel/yVel to the agent's position.
        void ForceMove()
        {
            this->MoveSafePT(this->Xpt() + m_xVel, this->Ypt() + m_yVel);
        }

        // Models cell division: places a new daughter agent divRadius away
        // from the parent along the axis in pScratchCoordArr (or, if pRand
        // is non-null, a random axis sampled into pScratchCoordArr first),
        // and moves the parent divRadius away in the opposite direction.
        // Returns the daughter agent. TSelf only defers naming the return
        // type until TGrid is complete; leave it defaulted.
        template <typename TSelf = TGrid>
        typename TSelf::AgentType *Divide(double  divRadius,
                                          double *pScratchCoordArr,
                                          Rand   *pRand)
        {
            if (pRand != nullptr)
            {
                pRand->RandomPointOnCircleEdge(divRadius, pScratchCoordArr);
            }
            double normSq = NormSquared(pScratchCoordArr[0], pScratchCoordArr[1]);
            if (normSq != (divRadius * divRadius))
            {
                double norm = std::sqrt(normSq);
                pScratchCoordArr[0] = pScratchCoordArr[0] * divRadius / norm;
                pScratchCoordArr[1] = pScratchCoordArr[1] * divRadius / norm;
            }
            typename TSelf::AgentType &child = this->Grid()->NewAgentPTSafe(
                this->Xpt() + pScratchCoordArr[0],
                this->Ypt() + pScratchCoordArr[1],
                this->Xpt(),
                this->Ypt());
            this->MoveSafePT(this->Xpt() - pScratchCoordArr[0], this->Ypt() - pScratchCoordArr[1]);
            return &child;
        }

        template <typename TSelf = TGrid>
        typename TSelf::AgentType *Divide(double  divRadius,
                                          Rand   &rand)
        {
            double scratchCoordArr[2] = {0, 0};
            return Divide(divRadius, scratchCoordArr, &rand);
        }

    private:
        double m_radius;
        double m_xVel;
        double m_yVel;
    };
}
