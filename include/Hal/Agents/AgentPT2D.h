#pragma once

#include "Hal/Agents/AgentBase.h"
#include "Hal/MathUtil.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Hal
{
    // Off-lattice 2D agent that can share a grid square with other agents,
    // ported from GridsAndAgents/AgentPT2D.java. Agents occupying the same
    // square form a doubly linked list (m_pNextSq/m_pPrevSq) threaded through
    // the grid's occupancy array — this bucket-list structure is what
    // AgentGrid2D::GetAgentsRad/GetAgentsRadApprox walk, and what Phase 4's
    // GPU neighbor search is expected to rebuild per-tick on the GPU (see
    // docs/DESIGN.md).
    //
    // TGrid must expose `using AgentType = <derived agent type>;` (see
    // AgentGrid2D.h).
    template <typename TGrid>
    class AgentPT2D : public AgentBase<TGrid>
    {
    public:
        explicit AgentPT2D(TGrid *pGrid)
            : AgentBase<TGrid>(pGrid),
              m_ptX(0),
              m_ptY(0),
              m_iSq(0),
              m_pNextSq(nullptr),
              m_pPrevSq(nullptr)
        {
        }

        ~AgentPT2D()
        {
        }

        double Xpt() const
        {
            return m_ptX;
        }

        double Ypt() const
        {
            return m_ptY;
        }

        int32_t Xsq() const
        {
            return static_cast<int32_t>(m_ptX);
        }

        int32_t Ysq() const
        {
            return static_cast<int32_t>(m_ptY);
        }

        int32_t Isq() const
        {
            return m_iSq;
        }

        void Setup(int32_t i)
        {
            m_ptX = this->m_pGrid->ItoX(i) + 0.5;
            m_ptY = this->m_pGrid->ItoY(i) + 0.5;
            m_iSq = i;
            AddSQ(i);
        }

        void Setup(double x,
                   double y)
        {
            m_ptX = x;
            m_ptY = y;
            m_iSq = this->m_pGrid->I(x, y);
            AddSQ(m_iSq);
        }

        void MoveSQ(int32_t newX,
                    int32_t newY)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            RemSQ();
            m_iSq = this->m_pGrid->I(newX, newY);
            AddSQ(m_iSq);
            m_ptX = newX + 0.5;
            m_ptY = newY + 0.5;
        }

        void MoveSQ(int32_t i)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            RemSQ();
            int32_t x = this->m_pGrid->ItoX(i);
            int32_t y = this->m_pGrid->ItoY(i);
            m_iSq = i;
            m_ptX = x + 0.5;
            m_ptY = y + 0.5;
            AddSQ(i);
        }

        void MovePT(double newX,
                    double newY)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            int32_t xIntNew = static_cast<int32_t>(newX);
            int32_t yIntNew = static_cast<int32_t>(newY);
            int32_t xIntOld = static_cast<int32_t>(m_ptX);
            int32_t yIntOld = static_cast<int32_t>(m_ptY);
            if ((xIntNew != xIntOld) || (yIntNew != yIntOld))
            {
                RemSQ();
                m_iSq = this->m_pGrid->I(xIntNew, yIntNew);
                AddSQ(m_iSq);
            }
            m_ptX = newX;
            m_ptY = newY;
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

        void MoveSafePT(double newX,
                        double newY)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            if (this->m_pGrid->In(newX, newY))
            {
                MovePT(newX, newY);
                return;
            }
            if (this->m_pGrid->IsWrapX())
            {
                newX = Wrap(newX, static_cast<double>(this->m_pGrid->Xdim()));
            }
            else if (!InDim(newX, static_cast<double>(this->m_pGrid->Xdim())))
            {
                newX = Xpt();
            }
            if (this->m_pGrid->IsWrapY())
            {
                newY = Wrap(newY, static_cast<double>(this->m_pGrid->Ydim()));
            }
            else if (!InDim(newY, static_cast<double>(this->m_pGrid->Ydim())))
            {
                newY = Ypt();
            }
            MovePT(newX, newY);
        }

        void Dispose()
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to dispose already dead agent");
            }
            RemSQ();
            this->m_pGrid->RemoveAgent(*static_cast<typename TGrid::AgentType *>(this));
        }

        // NOTE: templated on the pointer type (TPtr) rather than spelled
        // as `TGrid::AgentType*` — see the matching note in
        // AgentSQ2DUnstackable.h for why (TGrid is still incomplete at
        // the point this base class is first instantiated).
        template <typename TPtr>
        void GetAllOnSquare(std::vector<TPtr *> &putHere)
        {
            AgentPT2D<TGrid> *pToList = this;
            while (pToList != nullptr)
            {
                putHere.push_back(static_cast<TPtr *>(pToList));
                pToList = pToList->m_pNextSq;
            }
        }

        int32_t GetCountOnSquare() const
        {
            return this->m_pGrid->CountsArray()[m_iSq];
        }

        template <typename EvalFn>
        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength,
                        EvalFn   eval) const
        {
            return this->m_pGrid->MapHood(pHood, hoodLength, Xsq(), Ysq(), eval);
        }

        int32_t MapHood(int32_t *pHood,
                        int32_t  hoodLength) const
        {
            return this->m_pGrid->MapHood(pHood, hoodLength, Xsq(), Ysq());
        }

    private:
        void AddSQ(int32_t i)
        {
            typename TGrid::AgentType **ppGridArray = this->m_pGrid->GridArray();
            if (ppGridArray[i] != nullptr)
            {
                // Implicit derived-to-base conversion (TGrid::AgentType* to
                // AgentPT2D<TGrid>*), resolved here in the function body —
                // by call time TGrid::AgentType is complete.
                AgentPT2D<TGrid> *pOccupant = ppGridArray[i];
                pOccupant->m_pPrevSq = this;
                m_pNextSq = pOccupant;
            }
            ppGridArray[i] = static_cast<typename TGrid::AgentType *>(this);
            this->m_pGrid->CountsArray()[i]++;
        }

        void RemSQ()
        {
            typename TGrid::AgentType **ppGridArray = this->m_pGrid->GridArray();
            if (ppGridArray[m_iSq] == this)
            {
                ppGridArray[m_iSq] = static_cast<typename TGrid::AgentType *>(m_pNextSq);
            }
            if (m_pNextSq != nullptr)
            {
                m_pNextSq->m_pPrevSq = m_pPrevSq;
            }
            if (m_pPrevSq != nullptr)
            {
                m_pPrevSq->m_pNextSq = m_pNextSq;
            }
            m_pPrevSq = nullptr;
            m_pNextSq = nullptr;
            this->m_pGrid->CountsArray()[m_iSq]--;
        }

        double m_ptX;
        double m_ptY;
        int32_t m_iSq;
        // NOTE: typed as AgentPT2D<TGrid>* (this class itself), not
        // TGrid::AgentType* — a self-referential pointer member is always
        // fine regardless of TGrid's completeness, whereas
        // `TGrid::AgentType*` as a field type would force eager lookup of
        // that nested type at class-template-instantiation time, before
        // TGrid (the concrete grid) is complete. Downcasts to
        // TGrid::AgentType* happen only inside function bodies (lazily
        // instantiated, by which point TGrid is complete) — see the CRTP
        // note in DESIGN.md.
        AgentPT2D<TGrid> *m_pNextSq;
        AgentPT2D<TGrid> *m_pPrevSq;
    };
}
