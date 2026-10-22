#pragma once

#include "Hal/Agents/AgentPT3D.h"
#include "Hal/Grids/AgentGrid3D.h"
#include "Hal/MathUtil.h"
#include "Hal/Rand.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace Hal
{
    // Spherical off-lattice agent (see SphericalAgent2D.h).
    template <typename TGrid>
    class SphericalAgent3D : public AgentPT3D<TGrid>
    {
    public:
        explicit SphericalAgent3D(TGrid *pGrid)
            : AgentPT3D<TGrid>(pGrid),
              m_radius(0),
              m_xVel(0),
              m_yVel(0),
              m_zVel(0)
        {
        }

        ~SphericalAgent3D()
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

        double ZVel() const
        {
            return m_zVel;
        }

        void SetVelocity(double xVel,
                         double yVel,
                         double zVel)
        {
            m_xVel = xVel;
            m_yVel = yVel;
            m_zVel = zVel;
        }

        void DefaultInit(double radius)
        {
            m_radius = radius;
            m_xVel   = 0;
            m_yVel   = 0;
            m_zVel   = 0;
        }

        // See SphericalAgent2D::SumForces.
        template <typename OverlapFn>
        double SumForces(double   interactionRad,
                         OverlapFn overlapFun)
        {
            std::vector<typename TGrid::AgentType *> candidates;
            this->Grid()->GetAgentsRadApprox(candidates, this->Xpt(), this->Ypt(), this->Zpt(), interactionRad);
            double sum = 0;
            for (typename TGrid::AgentType *pOther : candidates)
            {
                if (pOther != this)
                {
                    double xComp = this->Grid()->DispX(this->Xpt(), pOther->Xpt());
                    double yComp = this->Grid()->DispY(this->Ypt(), pOther->Ypt());
                    double zComp = this->Grid()->DispZ(this->Zpt(), pOther->Zpt());
                    double dist  = Norm(xComp, yComp, zComp);
                    if ((dist < interactionRad) && (dist > 0.0))
                    {
                        double touchDist = (m_radius + pOther->Radius()) - dist;
                        double force     = overlapFun(touchDist, *pOther);
                        m_xVel -= (xComp / dist) * force;
                        m_yVel -= (yComp / dist) * force;
                        m_zVel -= (zComp / dist) * force;
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
        // AgentGrid3D::GetAgentsRad queried at this agent's position.
        template <typename TNeighbor, typename OverlapFn>
        double SumForces(const std::vector<TNeighbor *>             &neighbors,
                         const std::vector<AgentDisplacementInfo3D> &displacementInfo,
                         OverlapFn                                   overlapFun)
        {
            double sum = 0;
            for (size_t i = 0; i < neighbors.size(); i++)
            {
                TNeighbor                     *pOther = neighbors[i];
                const AgentDisplacementInfo3D &info = displacementInfo[i];
                if ((pOther != this) && (info.dist > 0.0))
                {
                    double touchDist = (m_radius + pOther->Radius()) - info.dist;
                    double force     = overlapFun(touchDist, *pOther);
                    m_xVel -= (info.dispX / info.dist) * force;
                    m_yVel -= (info.dispY / info.dist) * force;
                    m_zVel -= (info.dispZ / info.dist) * force;
                    if (force > 0)
                    {
                        sum += std::abs(force);
                    }
                }
            }
            return sum;
        }

        void ApplyFriction(double frictionConst)
        {
            m_xVel *= frictionConst;
            m_yVel *= frictionConst;
            m_zVel *= frictionConst;
        }

        void CapVelocity(double maxVel)
        {
            double maxVelSq = maxVel * maxVel;
            double normSq   = NormSquared(m_xVel, m_yVel, m_zVel);
            if (normSq > maxVelSq)
            {
                double convFactor = maxVel / std::sqrt(normSq);
                m_xVel *= convFactor;
                m_yVel *= convFactor;
                m_zVel *= convFactor;
            }
        }

        void ForceMove()
        {
            this->MoveSafePT(this->Xpt() + m_xVel, this->Ypt() + m_yVel, this->Zpt() + m_zVel);
        }

        // See SphericalAgent2D::Divide. With pRand null, the given axis is
        // rescaled to length divRadius.
        template <typename TSelf = TGrid>
        typename TSelf::AgentType *Divide(double  divRadius,
                                          double *pScratchCoordArr,
                                          Rand   *pRand)
        {
            if (pRand != nullptr)
            {
                pRand->RandomPointOnSphereEdge(divRadius, pScratchCoordArr);
            }
            else
            {
                double normSq = NormSquared(pScratchCoordArr[0], pScratchCoordArr[1], pScratchCoordArr[2]);
                if (normSq != (divRadius * divRadius))
                {
                    double norm = std::sqrt(normSq);
                    pScratchCoordArr[0] = pScratchCoordArr[0] * divRadius / norm;
                    pScratchCoordArr[1] = pScratchCoordArr[1] * divRadius / norm;
                    pScratchCoordArr[2] = pScratchCoordArr[2] * divRadius / norm;
                }
            }
            typename TSelf::AgentType &child = this->Grid()->NewAgentPTSafe(
                this->Xpt() + pScratchCoordArr[0],
                this->Ypt() + pScratchCoordArr[1],
                this->Zpt() + pScratchCoordArr[2],
                this->Xpt(),
                this->Ypt(),
                this->Zpt());
            this->MoveSafePT(
                this->Xpt() - pScratchCoordArr[0],
                this->Ypt() - pScratchCoordArr[1],
                this->Zpt() - pScratchCoordArr[2]);
            return &child;
        }

        template <typename TSelf = TGrid>
        typename TSelf::AgentType *Divide(double divRadius,
                                          Rand   &rand)
        {
            double scratchCoordArr[3] = {0, 0, 0};
            return Divide(divRadius, scratchCoordArr, &rand);
        }

    private:
        double m_radius;
        double m_xVel;
        double m_yVel;
        double m_zVel;
    };
}
