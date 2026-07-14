#pragma once

#include "Hal/Rand.h"

#include <cstdint>
#include <deque>
#include <vector>

namespace Hal
{
    // Agent object pool with dead-agent free-list and in-place recycling,
    // ported from HAL's GridsAndAgents/InternalGridAgentList.java.
    //
    // The Java version constructs new agents via reflection
    // (`Class<T>.getDeclaredConstructors()[0].newInstance()`) because
    // generics are erased at runtime. TAgent here is a compile-time
    // template parameter, so new agents are constructed directly via
    // `TAgent(TGrid*)` — no reflection needed (see docs/DESIGN.md).
    //
    // Agent objects live in `m_storage`, a std::deque so that pointers
    // into it stay valid across growth (deque never invalidates
    // references on push_back, unlike std::vector) — this replaces
    // Java's ArrayList<T>-of-references, which has the same stability
    // property for a different reason (the array holds references, not
    // objects inline).
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

        void RemoveAgent(TAgent &agent)
        {
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

        // Iterates alive agents, skipping any agent born during the
        // current iteration pass (stateID versioning) — matching Java's
        // AgentListIterator, which makes birth-during-iteration safe by
        // construction rather than by convention.
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

        // NOTE: begin()/end() are lowercase, breaking from the project's
        // CamelCase-for-functions convention, because C++'s range-based
        // for loop is defined to call container.begin()/.end() by that
        // exact spelling — this is a language protocol requirement, not a
        // style choice, so it's exempt from coding_standards.md here.
        //
        // Also simplified relative to Java's InternalGridAgentList, which
        // only increments stateID when not already inside an outer
        // iteration (to let nested/reentrant iteration share one pass).
        // Reentrant iteration isn't needed yet at this milestone, so this
        // always increments; revisit if a model needs nested iteration.
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
