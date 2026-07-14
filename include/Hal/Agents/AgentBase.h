#pragma once

#include <cstdint>

namespace Hal
{
    // Bookkeeping shared by every agent type, ported from
    // GridsAndAgents/AgentBase.java. Collapses AgentBase +
    // AgentBaseSpatial's non-spatial bookkeeping into one class — the
    // original's deeper hierarchy existed partly to support Java's
    // reflection-based construction and partly to share code across many
    // agent shapes; since TGrid/TAgent are compile-time template
    // parameters here (see InternalAgentPool.h), Age()/Dispose() don't
    // need virtual dispatch the way Java's abstract methods did (a given
    // AgentGrid only ever holds exactly one concrete agent type).
    template <typename TGrid>
    class AgentBase
    {
    public:
        explicit AgentBase(TGrid *pGrid)
            : m_pGrid(pGrid),
              m_stateId(0),
              m_iList(0),
              m_birthTick(0),
              m_alive(false)
        {
        }

        ~AgentBase()
        {
        }

        bool IsAlive() const
        {
            return m_alive;
        }

        int32_t BirthTick() const
        {
            return m_birthTick;
        }

        int32_t Age() const
        {
            return m_pGrid->GetTick() - m_birthTick;
        }

        TGrid *Grid() const
        {
            return m_pGrid;
        }

        // Pool bookkeeping — called by InternalAgentPool, not model code.
        void SetAlive(bool alive)
        {
            m_alive = alive;
        }

        void SetBirthTick(int32_t tick)
        {
            m_birthTick = tick;
        }

        void SetStateId(int64_t stateId)
        {
            m_stateId = stateId;
        }

        int64_t StateId() const
        {
            return m_stateId;
        }

        void SetIList(int32_t iList)
        {
            m_iList = iList;
        }

        int32_t IList() const
        {
            return m_iList;
        }

    protected:
        TGrid *m_pGrid;

    private:
        int64_t m_stateId;
        int32_t m_iList;
        int32_t m_birthTick;
        bool m_alive;
    };
}
