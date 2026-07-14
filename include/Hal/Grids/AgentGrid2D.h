#pragma once

#include "Hal/Agents/InternalAgentPool.h"
#include "Hal/Grids/GridGeometry2D.h"
#include "Hal/MathUtil.h"
#include "Hal/Rand.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace Hal
{
    // Per-found-agent distance/displacement record filled by the
    // displacement-info GetAgentsRad overloads: entry i describes
    // retAgentList[i] (Java's double[3] of [dist, xDisp, yDisp]).
    // Displacements are from the query point to the agent, wrap-aware.
    struct AgentDisplacementInfo2D
    {
        double dist;
        double dispX;
        double dispY;
    };

    // On-lattice/off-lattice 2D agent grid, ported from
    // GridsAndAgents/AgentGrid2D.java. Supports both the unstackable
    // single-agent-per-square path (AgentSQ2DUnstackable) and the
    // stackable, off-lattice, linked-list-per-cell path (AgentPT2D) — the
    // grid itself is agnostic to which; occupancy semantics come entirely
    // from which agent base class TAgent derives from, exactly as in Java.
    //
    // TDerived is the concrete grid class deriving from this one (CRTP),
    // e.g. `class GOLGrid : public AgentGrid2D<GOLAgent, GOLGrid>`. This is
    // needed because agent types are written against their *concrete* grid
    // (to reach grid-specific fields/methods, exactly like Java model code
    // does — e.g. `G.mooreHood`), not the generic base — so the agent
    // pool must construct new agents as `TAgent(TDerived*)`, not
    // `TAgent(AgentGrid2D<TAgent,TDerived>*)`.
    //
    // TAgent must derive from AgentSQ2DUnstackable<TDerived> or
    // AgentPT2D<TDerived>, and be constructible as TAgent(TDerived*).
    //
    template <typename TAgent, typename TDerived>
    class AgentGrid2D : public GridGeometry2D
    {
    public:
        using AgentType = TAgent;

        AgentGrid2D(int32_t xDim,
                    int32_t yDim)
            : AgentGrid2D(xDim, yDim, false, false)
        {
        }

        AgentGrid2D(int32_t xDim,
                    int32_t yDim,
                    bool    wrapX,
                    bool    wrapY)
            : GridGeometry2D(xDim, yDim, wrapX, wrapY),
              m_tick(0),
              m_grid(static_cast<size_t>(Length()), nullptr),
              m_counts(static_cast<size_t>(Length()), 0),
              m_agents(static_cast<TDerived *>(this))
        {
        }

        ~AgentGrid2D()
        {
        }

        TAgent *GetAgent(int32_t x,
                         int32_t y) const
        {
            return m_grid[I(x, y)];
        }

        TAgent *GetAgent(int32_t i) const
        {
            return m_grid[i];
        }

        TAgent *GetAgentSafe(int32_t x,
                             int32_t y) const
        {
            if (!In(x, y))
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
            }
            return m_grid[I(x, y)];
        }

        TAgent &NewAgentSQ(int32_t x,
                           int32_t y)
        {
            TAgent &newAgent = m_agents.GetNewAgent(m_tick);
            newAgent.Setup(x, y);
            return newAgent;
        }

        TAgent &NewAgentSQ(int32_t i)
        {
            TAgent &newAgent = m_agents.GetNewAgent(m_tick);
            newAgent.Setup(i);
            return newAgent;
        }

        TAgent &NewAgentPT(double x,
                           double y)
        {
            TAgent &newAgent = m_agents.GetNewAgent(m_tick);
            newAgent.Setup(x, y);
            return newAgent;
        }

        // Same as NewAgentPT, but applies wraparound (or falls back to
        // (fallbackX,fallbackY)) if (x,y) is out of bounds.
        TAgent &NewAgentPTSafe(double x,
                               double y,
                               double fallbackX,
                               double fallbackY)
        {
            if (In(x, y))
            {
                return NewAgentPT(x, y);
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
            return NewAgentPT(x, y);
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

        // Number of agents occupying square i (walks the square's stack for
        // stackable/off-lattice agent types; 0 or 1 for unstackable ones).
        int32_t PopAt(int32_t i) const
        {
            TAgent *pAgent = GetAgent(i);
            return (pAgent == nullptr) ? 0 : pAgent->GetCountOnSquare();
        }

        int32_t PopAt(int32_t x,
                      int32_t y) const
        {
            return PopAt(I(x, y));
        }

        void CleanAgents()
        {
            m_agents.CleanAgents();
        }

        void ShuffleAgents(Rand &rand)
        {
            m_agents.ShuffleAgents(rand);
        }

        // Agent-pool serialization. The reflection-free replacement for
        // Java's Serializable agent grids (see docs/DESIGN.md): the model
        // supplies a per-agent Save/Load callback for its own agent fields;
        // this writes the population and each agent's square index around it.
        // Templated on the writer/reader type so this header stays decoupled
        // from Hal::IO. LoadAgents assumes an empty grid of matching
        // dimensions (typically freshly constructed).
        //   saveAgent: void(TAgent &agent, Writer &writer)
        //   loadAgent: void(TAgent &agent, Reader &reader)
        template <typename Writer, typename SaveFn>
        void SaveAgents(Writer &writer,
                        SaveFn  saveAgent)
        {
            writer.WriteInt32(Pop());
            for (TAgent &agent : *this)
            {
                writer.WriteInt32(agent.Isq());
                saveAgent(agent, writer);
            }
        }

        template <typename Reader, typename LoadFn>
        void LoadAgents(Reader &reader,
                        LoadFn  loadAgent)
        {
            int32_t pop = reader.ReadInt32();
            for (int32_t k = 0; k < pop; k++)
            {
                int32_t iSq = reader.ReadInt32();
                TAgent &agent = NewAgentSQ(iSq);
                loadAgent(agent, reader);
            }
        }

        // Like MapHood, but keeps only the empty (unoccupied) neighbor
        // squares — the common "where can I divide into?" query. Returns the
        // count written into pHood's leading scratch slots.
        int32_t MapEmptyHood(int32_t *pHood,
                             int32_t  hoodLength,
                             int32_t  centerX,
                             int32_t  centerY) const
        {
            return this->MapHood(pHood, hoodLength, centerX, centerY,
                                 [this](int32_t i, int32_t, int32_t) { return GetAgent(i) == nullptr; });
        }

        int32_t MapEmptyHood(int32_t *pHood,
                             int32_t  hoodLength,
                             int32_t  centerI) const
        {
            return MapEmptyHood(pHood, hoodLength, this->ItoX(centerI), this->ItoY(centerI));
        }

        // Like MapHood, but keeps only the occupied neighbor squares.
        int32_t MapOccupiedHood(int32_t *pHood,
                                int32_t  hoodLength,
                                int32_t  centerX,
                                int32_t  centerY) const
        {
            return this->MapHood(pHood, hoodLength, centerX, centerY,
                                 [this](int32_t i, int32_t, int32_t) { return GetAgent(i) != nullptr; });
        }

        int32_t MapOccupiedHood(int32_t *pHood,
                                int32_t  hoodLength,
                                int32_t  centerI) const
        {
            return MapOccupiedHood(pHood, hoodLength, this->ItoX(centerI), this->ItoY(centerI));
        }

        // NOTE: lowercase begin()/end() to satisfy C++'s range-for
        // protocol (see the fuller note in InternalAgentPool.h) — exempt
        // from the CamelCase-for-functions convention for that reason.
        typename InternalAgentPool<TAgent, TDerived>::Iterator begin()
        {
            return m_agents.begin();
        }

        typename InternalAgentPool<TAgent, TDerived>::Iterator end()
        {
            return m_agents.end();
        }

        // Appends every agent occupying square (x,y) to putHere (walking
        // the cell's linked list for stackable/off-lattice agent types;
        // appends at most one for unstackable types).
        void GetAgents(std::vector<TAgent *> &putHere,
                       int32_t                x,
                       int32_t                y) const
        {
            TAgent *pAgent = GetAgent(x, y);
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
                           int32_t                x,
                           int32_t                y) const
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
            TAgent *pAgent = m_grid[I(x, y)];
            if (pAgent != nullptr)
            {
                pAgent->GetAllOnSquare(putHere);
            }
        }

        // Quickly gets all agents within rad of (x,y), but also includes
        // some agents further away than rad — an additional distance check
        // (GetAgentsRad, below) is needed to properly subset this group.
        // This is the raw cell-list bucket scan: it walks every grid
        // square whose bounding box overlaps the query circle.
        void GetAgentsRadApprox(std::vector<TAgent *> &retAgentList,
                                double                 x,
                                double                 y,
                                double                 rad) const
        {
            int32_t xStart = static_cast<int32_t>(std::floor(x - rad));
            int32_t xEnd = static_cast<int32_t>(std::ceil(x + rad));
            int32_t yStart = static_cast<int32_t>(std::floor(y - rad));
            int32_t yEnd = static_cast<int32_t>(std::ceil(y + rad));
            for (int32_t xSq = xStart; xSq < xEnd; xSq++)
            {
                for (int32_t ySq = yStart; ySq < yEnd; ySq++)
                {
                    int32_t retX = xSq;
                    int32_t retY = ySq;
                    bool inX = InDim(retX, Xdim());
                    bool inY = InDim(retY, Ydim());
                    if ((!IsWrapX() && !inX) || (!IsWrapY() && !inY))
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
                    GetAgents(retAgentList, retX, retY);
                }
            }
        }

        // Gets all agents within rad of (x,y) exactly, via the same
        // bucket-grid scan as GetAgentsRadApprox plus a true distance
        // check per candidate — this is the same cell-list neighbor
        // search the GPU compute path builds on directly.
        void GetAgentsRad(std::vector<TAgent *> &retAgentList,
                          double                 x,
                          double                 y,
                          double                 rad) const
        {
            double radSq = rad * rad;
            bool wrapX = IsWrapX();
            bool wrapY = IsWrapY();
            int32_t xDim = Xdim();
            int32_t yDim = Ydim();
            std::vector<TAgent *> candidates;
            int32_t xStart = static_cast<int32_t>(std::floor(x - rad));
            int32_t xEnd = static_cast<int32_t>(std::ceil(x + rad));
            int32_t yStart = static_cast<int32_t>(std::floor(y - rad));
            int32_t yEnd = static_cast<int32_t>(std::ceil(y + rad));
            for (int32_t xSq = xStart; xSq < xEnd; xSq++)
            {
                for (int32_t ySq = yStart; ySq < yEnd; ySq++)
                {
                    int32_t retX = xSq;
                    int32_t retY = ySq;
                    bool inX = InDim(retX, xDim);
                    bool inY = InDim(retY, yDim);
                    if ((!wrapX && !inX) || (!wrapY && !inY))
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
                    candidates.clear();
                    GetAgents(candidates, retX, retY);
                    for (TAgent *pAgent : candidates)
                    {
                        if (DistSquared(x, y, pAgent->Xpt(), pAgent->Ypt()) < radSq)
                        {
                            retAgentList.push_back(pAgent);
                        }
                    }
                }
            }
        }

        // EvalAgent-filtered variant of the raw bucket scan: same
        // over-inclusive square walk, keeping only agents the predicate
        // accepts (evalAgent(pAgent) -> bool, a template callable per
        // DESIGN.md's callback rule).
        template <typename EvalFn>
        void GetAgentsRadApprox(std::vector<TAgent *> &retAgentList,
                                double                 x,
                                double                 y,
                                double                 rad,
                                EvalFn                 evalAgent) const
        {
            std::vector<TAgent *> candidates;
            GetAgentsRadApprox(candidates, x, y, rad);
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
                          double                 x,
                          double                 y,
                          double                 rad,
                          EvalFn                 evalAgent) const
        {
            double radSq = rad * rad;
            std::vector<TAgent *> candidates;
            GetAgentsRadApprox(candidates, x, y, rad);
            for (TAgent *pAgent : candidates)
            {
                if ((DistSquared(x, y, pAgent->Xpt(), pAgent->Ypt()) < radSq) && evalAgent(pAgent))
                {
                    retAgentList.push_back(pAgent);
                }
            }
        }

        // Exact radius search that also records each found agent's
        // distance and wrap-aware displacement from the query point:
        // displacementInfo[i] describes retAgentList[i]. Like Java, the
        // info vector is padded so the indices line up even when
        // retAgentList already held entries from an earlier query.
        void GetAgentsRad(std::vector<TAgent *>                &retAgentList,
                          std::vector<AgentDisplacementInfo2D> &displacementInfo,
                          double                                x,
                          double                                y,
                          double                                rad) const
        {
            std::vector<TAgent *> candidates;
            GetAgentsRadApprox(candidates, x, y, rad);
            for (TAgent *pAgent : candidates)
            {
                double dispX = DispX(x, pAgent->Xpt());
                double dispY = DispY(y, pAgent->Ypt());
                double dist = Norm(dispX, dispY);
                if (dist < rad)
                {
                    if (displacementInfo.size() <= retAgentList.size())
                    {
                        displacementInfo.resize(retAgentList.size() + 1);
                    }
                    AgentDisplacementInfo2D &info = displacementInfo[retAgentList.size()];
                    info.dist = dist;
                    info.dispX = dispX;
                    info.dispY = dispY;
                    retAgentList.push_back(pAgent);
                }
            }
        }

        // Displacement-info search with an EvalAgent predicate on top.
        template <typename EvalFn>
        void GetAgentsRad(std::vector<TAgent *>                &retAgentList,
                          std::vector<AgentDisplacementInfo2D> &displacementInfo,
                          double                                x,
                          double                                y,
                          double                                rad,
                          EvalFn                                evalAgent) const
        {
            std::vector<TAgent *> candidates;
            GetAgentsRadApprox(candidates, x, y, rad);
            for (TAgent *pAgent : candidates)
            {
                double dispX = DispX(x, pAgent->Xpt());
                double dispY = DispY(y, pAgent->Ypt());
                double dist = Norm(dispX, dispY);
                if ((dist < rad) && evalAgent(pAgent))
                {
                    if (displacementInfo.size() <= retAgentList.size())
                    {
                        displacementInfo.resize(retAgentList.size() + 1);
                    }
                    AgentDisplacementInfo2D &info = displacementInfo[retAgentList.size()];
                    info.dist = dist;
                    info.dispX = dispX;
                    info.dispY = dispY;
                    retAgentList.push_back(pAgent);
                }
            }
        }

        // Internal: used by AgentSQ2DUnstackable/AgentPT2D to read/write
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
        int32_t m_tick;
        std::vector<TAgent *> m_grid;
        std::vector<int32_t> m_counts;
        InternalAgentPool<TAgent, TDerived> m_agents;
    };
}
