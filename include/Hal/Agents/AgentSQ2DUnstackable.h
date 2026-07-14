#pragma once

#include "Hal/Agents/AgentBase.h"
#include "Hal/MathUtil.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Hal
{
    // On-lattice 2D agent with at most one agent per grid square, ported
    // from GridsAndAgents/AgentSQ2Dunstackable.java. TGrid must expose
    // `using AgentType = <derived agent type>;` (see AgentGrid2D.h) so
    // AddSQ/RemSQ can address the grid's occupancy array with the correct
    // concrete type.
    template <typename TGrid>
    class AgentSQ2DUnstackable : public AgentBase<TGrid>
    {
    public:
        explicit AgentSQ2DUnstackable(TGrid *pGrid)
            : AgentBase<TGrid>(pGrid),
              m_xSq(0),
              m_ySq(0),
              m_iSq(0)
        {
        }

        ~AgentSQ2DUnstackable()
        {
        }

        int32_t Xsq() const
        {
            return m_xSq;
        }

        int32_t Ysq() const
        {
            return m_ySq;
        }

        double Xpt() const
        {
            return m_xSq + 0.5;
        }

        double Ypt() const
        {
            return m_ySq + 0.5;
        }

        int32_t Isq() const
        {
            return m_iSq;
        }

        void Setup(int32_t i)
        {
            m_xSq = this->m_pGrid->ItoX(i);
            m_ySq = this->m_pGrid->ItoY(i);
            m_iSq = i;
            AddSQ(i);
        }

        void Setup(int32_t x,
                   int32_t y)
        {
            m_xSq = x;
            m_ySq = y;
            m_iSq = this->m_pGrid->I(x, y);
            AddSQ(m_iSq);
        }

        void MoveSQ(int32_t i)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            m_xSq = this->m_pGrid->ItoX(i);
            m_ySq = this->m_pGrid->ItoY(i);
            this->m_pGrid->GridArray()[m_iSq] = nullptr;
            m_iSq = i;
            AddSQ(i);
        }

        void MoveSQ(int32_t x,
                    int32_t y)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            int32_t iNewPos = this->m_pGrid->I(x, y);
            RemSQ();
            AddSQ(iNewPos);
            m_xSq = x;
            m_ySq = y;
            m_iSq = iNewPos;
        }

        void MoveSafeSQ(int32_t newX,
                        int32_t newY)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            if (this->m_pGrid->In(newX, newY))
            {
                MoveSQ(newX, newY);
                return;
            }
            if (this->m_pGrid->IsWrapX())
            {
                newX = Wrap(newX, this->m_pGrid->Xdim());
            }
            else if (!InDim(newX, this->m_pGrid->Xdim()))
            {
                newX = Xsq();
            }
            if (this->m_pGrid->IsWrapY())
            {
                newY = Wrap(newY, this->m_pGrid->Ydim());
            }
            else if (!InDim(newY, this->m_pGrid->Ydim()))
            {
                newY = Ysq();
            }
            MoveSQ(newX, newY);
        }

        void SwapPosition(AgentSQ2DUnstackable<TGrid> &other)
        {
            if (!this->IsAlive() || !other.IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            if (other.m_pGrid != this->m_pGrid)
            {
                throw std::runtime_error("can't swap positions between agents on different grids");
            }
            other.RemSQ();
            RemSQ();
            int32_t xThis = m_xSq;
            int32_t yThis = m_ySq;
            int32_t iThis = m_iSq;
            m_xSq = other.m_xSq;
            m_ySq = other.m_ySq;
            m_iSq = other.m_iSq;
            other.m_xSq = xThis;
            other.m_ySq = yThis;
            other.m_iSq = iThis;
            other.AddSQ(other.m_iSq);
            AddSQ(m_iSq);
        }

        void Dispose()
        {
            RemSQ();
            this->m_pGrid->RemoveAgent(*static_cast<typename TGrid::AgentType *>(this));
        }

        template <typename EvalFn>
        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        EvalFn   eval) const
        {
            return this->m_pGrid->MapHood(pHood, hoodLength, m_xSq, m_ySq, eval);
        }

        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength) const
        {
            return this->m_pGrid->MapHood(pHood, hoodLength, m_xSq, m_ySq);
        }

        // Duck-typed to match AgentPT2D's linked-list walk, so
        // AgentGrid2D::GetAgents/GetAgentsRad work unchanged regardless of
        // which agent base TAgent derives from.
        //
        // NOTE: templated on the pointer type (TPtr), deduced from the
        // call site, rather than spelled as `TGrid::AgentType*` directly —
        // referencing that nested type in a member *declaration* forces
        // eager lookup at class-template-instantiation time, which is too
        // early: TGrid (the concrete grid) is still incomplete at the
        // point an agent class first instantiates this base (see the CRTP
        // note in DESIGN.md). A template method's signature is only
        // resolved when actually called, by which point TGrid is complete.
        template <typename TPtr>
        void GetAllOnSquare(std::vector<TPtr *> &putHere)
        {
            putHere.push_back(static_cast<TPtr *>(this));
        }

        int32_t GetCountOnSquare() const
        {
            return 1;
        }

    private:
        void AddSQ(int32_t i)
        {
            if (this->m_pGrid->GridArray()[i] != nullptr)
            {
                throw std::runtime_error("adding multiple unstackable agents to the same square");
            }
            this->m_pGrid->GridArray()[i] = static_cast<typename TGrid::AgentType *>(this);
        }

        void RemSQ()
        {
            this->m_pGrid->GridArray()[m_iSq] = nullptr;
        }

        int32_t m_xSq;
        int32_t m_ySq;
        int32_t m_iSq;
    };
}
