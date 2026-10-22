#pragma once

#include "Hal/Agents/AgentBase.h"
#include "Hal/MathUtil.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Hal
{
    // Off-lattice 2D agent that can share a grid square with other agents.
    // Agents on the same square form a doubly linked list headed in the
    // grid's occupancy array; AgentGrid2D's radius queries walk these
    // lists. TGrid must expose `using AgentType = <derived agent type>;`.
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

        // See AgentSQ2DUnstackable::GetAllOnSquare.
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
        // Base-typed because TGrid::AgentType can't be named while TGrid is
        // incomplete.
        AgentPT2D<TGrid> *m_pNextSq;
        AgentPT2D<TGrid> *m_pPrevSq;
    };
}
