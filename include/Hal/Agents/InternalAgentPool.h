#pragma once

#include "Hal/Rand.h"

#include <cstdint>
#include <deque>
#include <stdexcept>
#include <vector>

namespace Hal
{
    // Agent pool with a dead-agent free list and in-place recycling. Agents
    // are constructed as TAgent(TGrid*) and stored in a std::deque, so
    // pointers to them stay valid as the pool grows.
    template <typename TAgent, typename TGrid>
    class InternalAgentPool
    {
    public:
        explicit InternalAgentPool(TGrid *pGrid)
            : m_pGrid(pGrid),
              m_iLastAlive(-1),
              m_pop(0),
              m_stateId(0)
        {
        }

        ~InternalAgentPool()
        {
        }

        TAgent &GetNewAgent(int32_t birthTick)
        {
            TAgent *pNewAgent = nullptr;
            if (!m_deads.empty())
            {
                pNewAgent = m_deads.back();
                m_deads.pop_back();
            }
            else if (static_cast<int32_t>(m_agents.size()) > (m_iLastAlive + 1))
            {
                m_iLastAlive++;
                pNewAgent = m_agents[m_iLastAlive];
            }
            else
            {
                m_storage.emplace_back(m_pGrid);
                pNewAgent = &m_storage.back();
                m_agents.push_back(pNewAgent);
                m_iLastAlive++;
                pNewAgent->SetIList(m_iLastAlive);
            }
            pNewAgent->SetAlive(true);
            pNewAgent->SetStateId(m_stateId);
            pNewAgent->SetBirthTick(birthTick);
            m_pop++;
            return *pNewAgent;
        }

        // Throws std::runtime_error if agent is already dead.
        void RemoveAgent(TAgent &agent)
        {
            if (!agent.IsAlive())
            {
                throw std::runtime_error("attempting to dispose already dead agent");
            }
            agent.SetAlive(false);
            m_deads.push_back(&agent);
            m_pop--;
        }

        int32_t Pop() const
        {
            return m_pop;
        }

        void ShuffleAgents(Rand &rand)
        {
            m_stateId++;
            for (int32_t iSwap1 = m_iLastAlive; iSwap1 > 0; iSwap1--)
            {
                int32_t iSwap2 = rand.Int(iSwap1 + 1);
                TAgent *pSwap1 = m_agents[iSwap1];
                TAgent *pSwap2 = m_agents[iSwap2];
                pSwap1->SetIList(iSwap2);
                pSwap2->SetIList(iSwap1);
                m_agents[iSwap2] = pSwap1;
                m_agents[iSwap1] = pSwap2;
            }
        }

        void CleanAgents()
        {
            m_stateId++;
            int32_t iSwap = m_iLastAlive;
            m_iLastAlive = m_pop - 1;
            while (!m_deads.empty() && (iSwap > m_iLastAlive))
            {
                TAgent *pDead = m_deads.back();
                m_deads.pop_back();
                int32_t iDead = pDead->IList();
                if (iDead <= m_iLastAlive)
                {
                    TAgent *pSwap = m_agents[iSwap];
                    while (!pSwap->IsAlive())
                    {
                        iSwap--;
                        if (iSwap <= m_iLastAlive)
                        {
                            m_deads.clear();
                            return;
                        }
                        pSwap = m_agents[iSwap];
                    }
                    pSwap->SetIList(iDead);
                    pDead->SetIList(iSwap);
                    m_agents[iDead] = pSwap;
                    m_agents[iSwap] = pDead;
                    iSwap--;
                }
            }
            m_deads.clear();
        }

        // Visits alive agents in list order, skipping agents born after the
        // iteration began (they carry a newer state id).
        class Iterator
        {
        public:
            Iterator(int64_t            stateId,
                     InternalAgentPool *pPool,
                     int32_t            startIndex)
                : m_stateId(stateId),
                  m_pPool(pPool),
                  m_index(startIndex),
                  m_pCurrent(nullptr)
            {
                Advance();
            }

            ~Iterator()
            {
            }

            bool operator!=(const Iterator &other) const
            {
                (void)other;
                return m_pCurrent != nullptr;
            }

            TAgent &operator*() const
            {
                return *m_pCurrent;
            }

            Iterator &operator++()
            {
                Advance();
                return *this;
            }

        private:
            void Advance()
            {
                m_pCurrent = nullptr;
                while (m_index <= m_pPool->m_iLastAlive)
                {
                    TAgent *pCandidate = m_pPool->m_agents[m_index];
                    m_index++;
                    if (pCandidate->IsAlive() && (pCandidate->StateId() < m_stateId))
                    {
                        m_pCurrent = pCandidate;
                        return;
                    }
                }
            }

            int64_t m_stateId;
            InternalAgentPool *m_pPool;
            int32_t m_index;
            TAgent *m_pCurrent;
        };

        // Lowercase for range-based for.
        Iterator begin()
        {
            m_stateId++;
            return Iterator(m_stateId, this, 0);
        }

        Iterator end()
        {
            return Iterator(m_stateId, this, m_iLastAlive + 1);
        }

    private:
        TGrid *m_pGrid;
        std::deque<TAgent> m_storage;
        std::vector<TAgent *> m_agents;
        std::vector<TAgent *> m_deads;
        int32_t m_iLastAlive;
        int32_t m_pop;
        int64_t m_stateId;
    };
}
