#pragma once

#include "Hal/Agents/InternalAgentPool.h"
#include "Hal/Grids/GridGeometry3D.h"
#include "Hal/MathUtil.h"
#include "Hal/Rand.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace Hal
{
    // Distance and wrap-aware displacement from a query point to a found
    // agent, filled by the displacement-info GetAgentsRad overloads.
    struct AgentDisplacementInfo3D
    {
        double dist;
        double dispX;
        double dispY;
        double dispZ;
    };

    // 3D agent grid (see AgentGrid2D.h).
    template <typename TAgent, typename TDerived>
    class AgentGrid3D : public GridGeometry3D
    {
    public:
        using AgentType = TAgent;

        AgentGrid3D(int32_t xDim,
                    int32_t yDim,
                    int32_t zDim)
            : AgentGrid3D(xDim, yDim, zDim, false, false, false)
        {
        }

        AgentGrid3D(int32_t xDim,
                    int32_t yDim,
                    int32_t zDim,
                    bool    wrapX,
                    bool    wrapY,
                    bool    wrapZ)
            : GridGeometry3D(xDim, yDim, zDim, wrapX, wrapY, wrapZ),
              m_tick(0),
              m_grid(static_cast<size_t>(Length()), nullptr),
              m_counts(static_cast<size_t>(Length()), 0),
              m_agents(static_cast<TDerived *>(this))
        {
        }

        ~AgentGrid3D()
        {
        }

        TAgent *GetAgent(int32_t x,
                         int32_t y,
                         int32_t z) const
        {
            return m_grid[I(x, y, z)];
        }

        TAgent *GetAgent(int32_t i) const
        {
            return m_grid[i];
        }

        TAgent *GetAgentSafe(int32_t x,
                             int32_t y,
                             int32_t z) const
        {
            if (!In(x, y, z))
            {
                if (IsWrapX())
                {
                    x = Wrap(x, Xdim());
                }
                else if (!InDim(x, Xdim()))
                {
                    return nullptr;
                }
                if (IsWrapY())
                {
                    y = Wrap(y, Ydim());
                }
                else if (!InDim(y, Ydim()))
                {
                    return nullptr;
                }
                if (IsWrapZ())
                {
                    z = Wrap(z, Zdim());
                }
                else if (!InDim(z, Zdim()))
                {
                    return nullptr;
                }
            }
            return m_grid[I(x, y, z)];
        }

        TAgent &NewAgentSQ(int32_t x,
                           int32_t y,
                           int32_t z)
        {
            TAgent &newAgent = m_agents.GetNewAgent(m_tick);
            newAgent.Setup(x, y, z);
            return newAgent;
        }

        TAgent &NewAgentSQ(int32_t i)
        {
            TAgent &newAgent = m_agents.GetNewAgent(m_tick);
            newAgent.Setup(i);
            return newAgent;
        }

        TAgent &NewAgentPT(double x,
                           double y,
                           double z)
        {
            TAgent &newAgent = m_agents.GetNewAgent(m_tick);
            newAgent.Setup(x, y, z);
            return newAgent;
        }

        TAgent &NewAgentPTSafe(double x,
                               double y,
                               double z,
                               double fallbackX,
                               double fallbackY,
                               double fallbackZ)
        {
            if (In(x, y, z))
            {
                return NewAgentPT(x, y, z);
            }
            if (IsWrapX())
            {
                x = Wrap(x, static_cast<double>(Xdim()));
            }
            else if (!InDim(x, static_cast<double>(Xdim())))
            {
                x = fallbackX;
            }
            if (IsWrapY())
            {
                y = Wrap(y, static_cast<double>(Ydim()));
            }
            else if (!InDim(y, static_cast<double>(Ydim())))
            {
                y = fallbackY;
            }
            if (IsWrapZ())
            {
                z = Wrap(z, static_cast<double>(Zdim()));
            }
            else if (!InDim(z, static_cast<double>(Zdim())))
            {
                z = fallbackZ;
            }
            return NewAgentPT(x, y, z);
        }

        void RemoveAgent(TAgent &agent)
        {
            m_agents.RemoveAgent(agent);
        }

        int32_t GetTick() const
        {
            return m_tick;
        }

        void IncTick()
        {
            m_tick++;
        }

        int32_t Pop() const
        {
            return m_agents.Pop();
        }

        // Number of agents occupying voxel i.
        int32_t PopAt(int32_t i) const
        {
            TAgent *pAgent = GetAgent(i);
            return (pAgent == nullptr) ? 0 : pAgent->GetCountOnSquare();
        }

        int32_t PopAt(int32_t x,
                      int32_t y,
                      int32_t z) const
        {
            return PopAt(I(x, y, z));
        }

        void CleanAgents()
        {
            m_agents.CleanAgents();
        }

        void ShuffleAgents(Rand &rand)
        {
            m_agents.ShuffleAgents(rand);
        }

        // Like MapHood, but keeps only the empty (unoccupied) neighbor
        // squares — the 3D counterpart of AgentGrid2D::MapEmptyHood.
        int32_t MapEmptyHood(int32_t *pHood,
                             int32_t  hoodLength,
                             int32_t  centerX,
                             int32_t  centerY,
                             int32_t  centerZ) const
        {
            return this->MapHood(pHood, hoodLength, centerX, centerY, centerZ,
                                 [this](int32_t i, int32_t, int32_t, int32_t) { return GetAgent(i) == nullptr; });
        }

        int32_t MapEmptyHood(int32_t *pHood,
                             int32_t  hoodLength,
                             int32_t  centerI) const
        {
            return MapEmptyHood(pHood, hoodLength, this->ItoX(centerI), this->ItoY(centerI), this->ItoZ(centerI));
        }

        // Lowercase for range-based for.
        typename InternalAgentPool<TAgent, TDerived>::Iterator begin()
        {
            return m_agents.begin();
        }

        typename InternalAgentPool<TAgent, TDerived>::Iterator end()
        {
            return m_agents.end();
        }

        void GetAgents(std::vector<TAgent *> &putHere,
                       int32_t                x,
                       int32_t                y,
                       int32_t                z) const
        {
            TAgent *pAgent = GetAgent(x, y, z);
            if (pAgent != nullptr)
            {
                pAgent->GetAllOnSquare(putHere);
            }
        }

        void GetAgents(std::vector<TAgent *> &putHere,
                       int32_t                i) const
        {
            TAgent *pAgent = m_grid[i];
            if (pAgent != nullptr)
            {
                pAgent->GetAllOnSquare(putHere);
            }
        }

        void GetAgentsSafe(std::vector<TAgent *> &putHere,
                           int32_t                 x,
                           int32_t                 y,
                           int32_t                 z) const
        {
            if (IsWrapX())
            {
                x = Wrap(x, Xdim());
            }
            else if (!InDim(x, Xdim()))
            {
                return;
            }
            if (IsWrapY())
            {
                y = Wrap(y, Ydim());
            }
            else if (!InDim(y, Ydim()))
            {
                return;
            }
            if (IsWrapZ())
            {
                z = Wrap(z, Zdim());
            }
            else if (!InDim(z, Zdim()))
            {
                return;
            }
            TAgent *pAgent = m_grid[I(x, y, z)];
            if (pAgent != nullptr)
            {
                pAgent->GetAllOnSquare(putHere);
            }
        }

        // Raw cell-list bucket scan over the query sphere's bounding box —
        // see AgentGrid2D::GetAgentsRadApprox for the caveat (includes
        // some agents further away than rad).
        void GetAgentsRadApprox(std::vector<TAgent *> &retAgentList,
                                double                  x,
                                double                  y,
                                double                  z,
                                double                  rad) const
        {
            int32_t xStart = static_cast<int32_t>(std::floor(x - rad));
            int32_t xEnd   = static_cast<int32_t>(std::ceil(x + rad));
            int32_t yStart = static_cast<int32_t>(std::floor(y - rad));
            int32_t yEnd   = static_cast<int32_t>(std::ceil(y + rad));
            int32_t zStart = static_cast<int32_t>(std::floor(z - rad));
            int32_t zEnd   = static_cast<int32_t>(std::ceil(z + rad));
            for (int32_t xSq = xStart; xSq < xEnd; xSq++)
            {
                for (int32_t ySq = yStart; ySq < yEnd; ySq++)
                {
                    for (int32_t zSq = zStart; zSq < zEnd; zSq++)
                    {
                        int32_t retX = xSq;
                        int32_t retY = ySq;
                        int32_t retZ = zSq;
                        bool    inX  = InDim(retX, Xdim());
                        bool    inY  = InDim(retY, Ydim());
                        bool    inZ  = InDim(retZ, Zdim());
                        if ((!IsWrapX() && !inX) || (!IsWrapY() && !inY) || (!IsWrapZ() && !inZ))
                        {
                            continue;
                        }
                        if (IsWrapX() && !inX)
                        {
                            retX = Wrap(retX, Xdim());
                        }
                        if (IsWrapY() && !inY)
                        {
                            retY = Wrap(retY, Ydim());
                        }
                        if (IsWrapZ() && !inZ)
                        {
                            retZ = Wrap(retZ, Zdim());
                        }
                        GetAgents(retAgentList, retX, retY, retZ);
                    }
                }
            }
        }

        // Gets all agents within rad of (x,y,z): the GetAgentsRadApprox
        // scan plus a wrap-aware distance check per candidate.
        void GetAgentsRad(std::vector<TAgent *> &retAgentList,
                          double                  x,
                          double                  y,
                          double                  z,
                          double                  rad) const
        {
            double  radSq = rad * rad;
            bool    wrapX = IsWrapX();
            bool    wrapY = IsWrapY();
            bool    wrapZ = IsWrapZ();
            int32_t xDim  = Xdim();
            int32_t yDim  = Ydim();
            int32_t zDim  = Zdim();
            std::vector<TAgent *> candidates;
            int32_t xStart = static_cast<int32_t>(std::floor(x - rad));
            int32_t xEnd   = static_cast<int32_t>(std::ceil(x + rad));
            int32_t yStart = static_cast<int32_t>(std::floor(y - rad));
            int32_t yEnd   = static_cast<int32_t>(std::ceil(y + rad));
            int32_t zStart = static_cast<int32_t>(std::floor(z - rad));
            int32_t zEnd   = static_cast<int32_t>(std::ceil(z + rad));
            for (int32_t xSq = xStart; xSq < xEnd; xSq++)
            {
                for (int32_t ySq = yStart; ySq < yEnd; ySq++)
                {
                    for (int32_t zSq = zStart; zSq < zEnd; zSq++)
                    {
                        int32_t retX = xSq;
                        int32_t retY = ySq;
                        int32_t retZ = zSq;
                        bool    inX  = InDim(retX, xDim);
                        bool    inY  = InDim(retY, yDim);
                        bool    inZ  = InDim(retZ, zDim);
                        if ((!wrapX && !inX) || (!wrapY && !inY) || (!wrapZ && !inZ))
                        {
                            continue;
                        }
                        if (wrapX && !inX)
                        {
                            retX = Wrap(retX, xDim);
                        }
                        if (wrapY && !inY)
                        {
                            retY = Wrap(retY, yDim);
                        }
                        if (wrapZ && !inZ)
                        {
                            retZ = Wrap(retZ, zDim);
                        }
                        candidates.clear();
                        GetAgents(candidates, retX, retY, retZ);
                        for (TAgent *pAgent : candidates)
                        {
                            if (DistSquared(x, y, z, pAgent->Xpt(), pAgent->Ypt(), pAgent->Zpt()) < radSq)
                            {
                                retAgentList.push_back(pAgent);
                            }
                        }
                    }
                }
            }
        }

        // GetAgentsRadApprox keeping only agents evalAgent(pAgent) accepts.
        template <typename EvalFn>
        void GetAgentsRadApprox(std::vector<TAgent *> &retAgentList,
                                double                  x,
                                double                  y,
                                double                  z,
                                double                  rad,
                                EvalFn                  evalAgent) const
        {
            std::vector<TAgent *> candidates;
            GetAgentsRadApprox(candidates, x, y, z, rad);
            for (TAgent *pAgent : candidates)
            {
                if (evalAgent(pAgent))
                {
                    retAgentList.push_back(pAgent);
                }
            }
        }

        // EvalAgent-filtered exact radius search: within rad AND accepted
        // by the predicate.
        template <typename EvalFn>
        void GetAgentsRad(std::vector<TAgent *> &retAgentList,
                          double                  x,
                          double                  y,
                          double                  z,
                          double                  rad,
                          EvalFn                  evalAgent) const
        {
            double radSq = rad * rad;
            std::vector<TAgent *> candidates;
            GetAgentsRadApprox(candidates, x, y, z, rad);
            for (TAgent *pAgent : candidates)
            {
                if ((DistSquared(x, y, z, pAgent->Xpt(), pAgent->Ypt(), pAgent->Zpt()) < radSq) && evalAgent(pAgent))
                {
                    retAgentList.push_back(pAgent);
                }
            }
        }

        // See AgentGrid2D's displacement-info GetAgentsRad.
        void GetAgentsRad(std::vector<TAgent *>                &retAgentList,
                          std::vector<AgentDisplacementInfo3D> &displacementInfo,
                          double                                x,
                          double                                y,
                          double                                z,
                          double                                rad) const
        {
            std::vector<TAgent *> candidates;
            GetAgentsRadApprox(candidates, x, y, z, rad);
            for (TAgent *pAgent : candidates)
            {
                double dispX = DispX(x, pAgent->Xpt());
                double dispY = DispY(y, pAgent->Ypt());
                double dispZ = DispZ(z, pAgent->Zpt());
                double dist = Norm(dispX, dispY, dispZ);
                if (dist < rad)
                {
                    if (displacementInfo.size() <= retAgentList.size())
                    {
                        displacementInfo.resize(retAgentList.size() + 1);
                    }
                    AgentDisplacementInfo3D &info = displacementInfo[retAgentList.size()];
                    info.dist = dist;
                    info.dispX = dispX;
                    info.dispY = dispY;
                    info.dispZ = dispZ;
                    retAgentList.push_back(pAgent);
                }
            }
        }

        // Displacement-info search with an EvalAgent predicate on top.
        template <typename EvalFn>
        void GetAgentsRad(std::vector<TAgent *>                &retAgentList,
                          std::vector<AgentDisplacementInfo3D> &displacementInfo,
                          double                                x,
                          double                                y,
                          double                                z,
                          double                                rad,
                          EvalFn                                evalAgent) const
        {
            std::vector<TAgent *> candidates;
            GetAgentsRadApprox(candidates, x, y, z, rad);
            for (TAgent *pAgent : candidates)
            {
                double dispX = DispX(x, pAgent->Xpt());
                double dispY = DispY(y, pAgent->Ypt());
                double dispZ = DispZ(z, pAgent->Zpt());
                double dist = Norm(dispX, dispY, dispZ);
                if ((dist < rad) && evalAgent(pAgent))
                {
                    if (displacementInfo.size() <= retAgentList.size())
                    {
                        displacementInfo.resize(retAgentList.size() + 1);
                    }
                    AgentDisplacementInfo3D &info = displacementInfo[retAgentList.size()];
                    info.dist = dist;
                    info.dispX = dispX;
                    info.dispY = dispY;
                    info.dispZ = dispZ;
                    retAgentList.push_back(pAgent);
                }
            }
        }

        // Internal: used by AgentSQ3DUnstackable/AgentPT3D to read/write
        // occupancy and per-cell agent counts.
        TAgent **GridArray()
        {
            return m_grid.data();
        }

        int32_t *CountsArray()
        {
            return m_counts.data();
        }

    private:
        int32_t               m_tick;
        std::vector<TAgent *> m_grid;
        std::vector<int32_t>  m_counts;
        InternalAgentPool<TAgent, TDerived> m_agents;
    };
}
