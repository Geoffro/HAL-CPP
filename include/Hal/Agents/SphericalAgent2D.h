#pragma once

#include "Hal/Agents/AgentPT2D.h"
#include "Hal/MathUtil.h"
#include "Hal/Rand.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace Hal
{
    // "Spherical cow" 2D physics agent — circular agents that repel or
    // attract each other via a pluggable overlap-force law, ported from
    // GridsAndAgents/SphericalAgent2D.java.
    //
    // NOTE: Java's zero-scratch-arg Divide() overload used a
    // ThreadLocal<double[]> to lazily allocate per-thread scratch space.
    // Since nothing here is threaded yet, and the scratch is only needed
    // for the duration of one call, the port just uses a local stack array
    // instead — simpler, and the ThreadLocal's whole purpose (avoiding a
    // shared, contended buffer across threads) doesn't apply to a purely
    // local variable.
    //
    // NOTE: the cross-grid SumForces overload (interacting with agents on
    // a *different* AgentGrid2D) and the GetFreshAgentSearchArr scratch-
    // buffer reuse pool are deferred — this covers the core force-law
    // mechanism against agents on the agent's own grid.
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
        // < 0 means they're separated by that distance. A positive
        // returned force repels the agent away from the overlap
        // direction; negative pulls it toward that direction. Alters
        // xVel/yVel for every other agent within interactionRad and
        // returns the sum of positive (repulsive) force magnitudes.
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
                    if (dist < interactionRad)
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

        // Same force law as the interactionRad overload above, but against
        // an already-computed neighbor list + per-neighbor displacement
        // info ([dist, xDisp, yDisp] per entry, matching
        // GridGeometry2D::GetAgentsRad's displacement-info output) rather
        // than performing its own radius search.
        template <typename TNeighbor, typename OverlapFn>
        double SumForces(const std::vector<TNeighbor *>           &neighbors,
                         const std::vector<std::array<double, 3>> &displacementInfo,
                         OverlapFn                                 overlapFun)
        {
            double sum = 0;
            for (size_t i = 0; i < neighbors.size(); i++)
            {
                TNeighbor *pOther = neighbors[i];
                if (pOther != this)
                {
                    const std::array<double, 3> &info = displacementInfo[i];
                    double dist = info[0];
                    double xComp = info[1];
                    double yComp = info[2];
                    double touchDist = (m_radius + pOther->Radius()) - dist;
                    double force = overlapFun(touchDist, *pOther);
                    m_xVel += (xComp / dist) * force;
                    m_yVel += (yComp / dist) * force;
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
        // Returns the daughter agent.
        //
        // NOTE: templated on TSelf (defaulted to TGrid, never meant to be
        // overridden by callers) purely so the return type isn't spelled
        // as `TGrid::AgentType*` directly in the declaration — same
        // incomplete-type pitfall as GetAllOnSquare (see AgentPT2D.h):
        // spelling it there would force eager lookup while TGrid is still
        // incomplete. Making Divide itself a template defers that lookup
        // to actual instantiation (i.e. when it's first called), by which
        // point TGrid is complete.
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
