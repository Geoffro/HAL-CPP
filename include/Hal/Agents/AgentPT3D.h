#pragma once

#include "Hal/Agents/AgentBase.h"
#include "Hal/MathUtil.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Hal
{
    // Off-lattice 3D agent that can share a voxel with other agents (see
    // AgentPT2D.h).
    template <typename TGrid>
    class AgentPT3D : public AgentBase<TGrid>
    {
    public:
        explicit AgentPT3D(TGrid *pGrid)
            : AgentBase<TGrid>(pGrid),
              m_ptX(0),
              m_ptY(0),
              m_ptZ(0),
              m_iSq(0),
              m_pNextSq(nullptr),
              m_pPrevSq(nullptr)
        {
        }

        ~AgentPT3D()
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

        double Zpt() const
        {
            return m_ptZ;
        }

        int32_t Xsq() const
        {
            return static_cast<int32_t>(m_ptX);
        }

        int32_t Ysq() const
        {
            return static_cast<int32_t>(m_ptY);
        }

        int32_t Zsq() const
        {
            return static_cast<int32_t>(m_ptZ);
        }

        int32_t Isq() const
        {
            return m_iSq;
        }

        void Setup(int32_t i)
        {
            m_ptX = this->m_pGrid->ItoX(i) + 0.5;
            m_ptY = this->m_pGrid->ItoY(i) + 0.5;
            m_ptZ = this->m_pGrid->ItoZ(i) + 0.5;
            m_iSq = i;
            AddSQ(i);
        }

        void Setup(double x,
                   double y,
                   double z)
        {
            m_ptX = x;
            m_ptY = y;
            m_ptZ = z;
            m_iSq = this->m_pGrid->I(x, y, z);
            AddSQ(m_iSq);
        }

        void MoveSQ(int32_t newX,
                    int32_t newY,
                    int32_t newZ)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            RemSQ();
            m_iSq = this->m_pGrid->I(newX, newY, newZ);
            AddSQ(m_iSq);
            m_ptX = newX + 0.5;
            m_ptY = newY + 0.5;
            m_ptZ = newZ + 0.5;
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
            int32_t z = this->m_pGrid->ItoZ(i);
            m_iSq = i;
            m_ptX = x + 0.5;
            m_ptY = y + 0.5;
            m_ptZ = z + 0.5;
            AddSQ(i);
        }

        void MovePT(double newX,
                    double newY,
                    double newZ)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            int32_t xIntNew = static_cast<int32_t>(newX);
            int32_t yIntNew = static_cast<int32_t>(newY);
            int32_t zIntNew = static_cast<int32_t>(newZ);
            int32_t xIntOld = static_cast<int32_t>(m_ptX);
            int32_t yIntOld = static_cast<int32_t>(m_ptY);
            int32_t zIntOld = static_cast<int32_t>(m_ptZ);
            if ((xIntNew != xIntOld) || (yIntNew != yIntOld) || (zIntNew != zIntOld))
            {
                RemSQ();
                m_iSq = this->m_pGrid->I(xIntNew, yIntNew, zIntNew);
                AddSQ(m_iSq);
            }
            m_ptX = newX;
            m_ptY = newY;
            m_ptZ = newZ;
        }

        void MoveSafeSQ(int32_t newX,
                        int32_t newY,
                        int32_t newZ)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            if (this->m_pGrid->In(newX, newY, newZ))
            {
                MoveSQ(newX, newY, newZ);
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
            if (this->m_pGrid->IsWrapZ())
            {
                newZ = Wrap(newZ, this->m_pGrid->Zdim());
            }
            else if (!InDim(newZ, this->m_pGrid->Zdim()))
            {
                newZ = Zsq();
            }
            MoveSQ(newX, newY, newZ);
        }

        void MoveSafePT(double newX,
                        double newY,
                        double newZ)
        {
            if (!this->IsAlive())
            {
                throw std::runtime_error("attempting to move dead agent");
            }
            if (this->m_pGrid->In(newX, newY, newZ))
            {
                MovePT(newX, newY, newZ);
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
            if (this->m_pGrid->IsWrapZ())
            {
                newZ = Wrap(newZ, static_cast<double>(this->m_pGrid->Zdim()));
            }
            else if (!InDim(newZ, static_cast<double>(this->m_pGrid->Zdim())))
            {
                newZ = Zpt();
            }
            MovePT(newX, newY, newZ);
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

        template <typename TPtr>
        void GetAllOnSquare(std::vector<TPtr *> &putHere)
        {
            AgentPT3D<TGrid> *pToList = this;
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
                        int32_t hoodLength,
                        EvalFn  eval) const
        {
            return this->m_pGrid->MapHood(pHood, hoodLength, Xsq(), Ysq(), Zsq(), eval);
        }

        int32_t MapHood(int32_t *pHood,
                        int32_t hoodLength) const
        {
            return this->m_pGrid->MapHood(pHood, hoodLength, Xsq(), Ysq(), Zsq());
        }

    private:
        void AddSQ(int32_t i)
        {
            typename TGrid::AgentType **ppGridArray = this->m_pGrid->GridArray();
            if (ppGridArray[i] != nullptr)
            {
                AgentPT3D<TGrid> *pOccupant = ppGridArray[i];
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
        double m_ptZ;
        int32_t m_iSq;
        AgentPT3D<TGrid> *m_pNextSq;
        AgentPT3D<TGrid> *m_pPrevSq;
    };
}
