#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Hal
{
    // Count-based population grid core (TCount is int32_t or int64_t).
    // Changes accumulate in a delta buffer that Update() commits. Iteration
    // adapts to occupancy: when at most 0.1% of cells are occupied it walks
    // a linked list of occupied indices instead of scanning the grid, and
    // returns to scanning once occupancy reaches 1%. The total population is
    // kept as int64_t.
    template <typename TCount>
    class PopulationGridBase
    {
    public:
        explicit PopulationGridBase(int32_t length)
            : m_agents(static_cast<size_t>(length), TCount{}),
              m_deltas(static_cast<size_t>(length), TCount{}),
              m_occupiedArea(0),
              m_pop(0),
              m_updateCt(0),
              m_length(length),
              m_usingSparseIndices(false),
              m_firstLiveIndex(kUnlinked),
              m_firstLiveDelta(kUnlinked)
        {
        }

        ~PopulationGridBase()
        {
        }

        void Add(int32_t i,
                 TCount  val)
        {
            if (val != TCount{})
            {
                m_deltas[i] += val;
                if (m_usingSparseIndices)
                {
                    LinkDeltaIndex(i);
                }
            }
        }

        void Set(int32_t i,
                 TCount  val)
        {
            Add(i, val - m_agents[i]);
        }

        void SetAll(TCount val)
        {
            for (int32_t i = 0; i < m_length; i++)
            {
                Add(i, val - m_agents[i]);
            }
        }

        int64_t Pop() const
        {
            return m_pop;
        }

        int32_t OccupiedArea() const
        {
            return m_occupiedArea;
        }

        int32_t UpdateCt() const
        {
            return m_updateCt;
        }

        int32_t Length() const
        {
            return m_length;
        }

        TCount Get(int32_t i) const
        {
            return m_agents[i];
        }

        void Move(int32_t iFrom,
                  int32_t iTo,
                  TCount  val)
        {
            Add(iFrom, -val);
            Add(iTo, val);
        }

        void Move(int32_t                     iFrom,
                  int32_t                     iTo,
                  PopulationGridBase<TCount> &gridTo,
                  TCount                      val)
        {
            Add(iFrom, -val);
            gridTo.Add(iTo, val);
        }

        // Calls action(i, count) for every occupied index. Adding during
        // the walk is fine (it only touches deltas); throws if action calls
        // Update().
        template <typename ActionFn>
        void ApplyOccupied(ActionFn action)
        {
            int32_t updateId = m_updateCt;
            if (m_usingSparseIndices)
            {
                int32_t i = m_firstLiveIndex;
                while (i >= 0)
                {
                    action(i, m_agents[i]);
                    i = m_nextLiveIndex[i];
                }
            }
            else
            {
                for (int32_t i = 0; i < m_length; i++)
                {
                    if (m_agents[i] != TCount{})
                    {
                        action(i, m_agents[i]);
                    }
                }
            }
            if (updateId != m_updateCt)
            {
                throw std::runtime_error("update during apply iteration not permitted");
            }
        }

        void Update()
        {
            m_updateCt++;
            if (m_usingSparseIndices)
            {
                int32_t i    = m_firstLiveDelta;
                m_firstLiveDelta = kUnlinked;
                while (i >= 0)
                {
                    TCount prev = m_agents[i];
                    m_agents[i] += m_deltas[i];
                    m_pop += m_deltas[i];
                    m_deltas[i] = TCount{};
                    if ((prev == TCount{}) && (m_agents[i] != TCount{}))
                    {
                        AddPopIndex(i);
                    }
                    else if ((prev != TCount{}) && (m_agents[i] == TCount{}))
                    {
                        RemovePopIndex(i);
                    }
                    if (m_agents[i] < TCount{})
                    {
                        throw std::runtime_error("population is below zero (overflow/underflow)");
                    }
                    int32_t j = m_nextLiveDelta[i];
                    m_nextLiveDelta[i] = kUnlinked;
                    i = j;
                }
            }
            else
            {
                for (int32_t i = 0; i < m_length; i++)
                {
                    TCount prev = m_agents[i];
                    m_agents[i] += m_deltas[i];
                    m_pop += m_deltas[i];
                    m_deltas[i] = TCount{};
                    if ((prev == TCount{}) && (m_agents[i] != TCount{}))
                    {
                        m_occupiedArea++;
                    }
                    else if ((prev != TCount{}) && (m_agents[i] == TCount{}))
                    {
                        m_occupiedArea--;
                    }
                    if (m_agents[i] < TCount{})
                    {
                        throw std::runtime_error("population is below zero (overflow/underflow)");
                    }
                }
            }
            if (!m_usingSparseIndices && (m_occupiedArea <= static_cast<int32_t>(m_length * 0.001)))
            {
                SetupSparseIndices();
            }
            if (m_usingSparseIndices && (m_occupiedArea >= static_cast<int32_t>(m_length * 0.01)))
            {
                m_usingSparseIndices = false;
            }
        }

        // Switches to sparse iteration now, rebuilding the occupied-index and
        // pending-delta lists. Update() calls this when occupancy falls to
        // the sparse threshold and may switch back to dense iteration.
        void SetupSparseIndices()
        {
            if (m_nextLiveIndex.empty())
            {
                m_nextLiveIndex.assign(static_cast<size_t>(m_length), kUnlinked);
                m_prevLiveIndex.assign(static_cast<size_t>(m_length), kUnlinked);
                m_nextLiveDelta.assign(static_cast<size_t>(m_length), kUnlinked);
            }
            else
            {
                std::fill(m_nextLiveIndex.begin(), m_nextLiveIndex.end(), kUnlinked);
                std::fill(m_prevLiveIndex.begin(), m_prevLiveIndex.end(), kUnlinked);
                std::fill(m_nextLiveDelta.begin(), m_nextLiveDelta.end(), kUnlinked);
            }
            m_firstLiveIndex = kUnlinked;
            m_firstLiveDelta = kUnlinked;

            for (int32_t i = 0; i < m_length; i++)
            {
                if (m_agents[i] > TCount{})
                {
                    LinkPopIndex(i);
                }
                if (m_deltas[i] != TCount{})
                {
                    LinkDeltaIndex(i);
                }
            }
            m_usingSparseIndices = true;
        }

        void AddTo(TCount *pDest) const
        {
            for (int32_t i = 0; i < m_length; i++)
            {
                pDest[i] += m_agents[i];
            }
        }

        void CopyTo(TCount *pDest) const
        {
            for (int32_t i = 0; i < m_length; i++)
            {
                pDest[i] = m_agents[i];
            }
        }

        // Iterates occupied indices (dense scan or sparse list walk).
        class Iterator
        {
        public:
            Iterator(PopulationGridBase *pGrid,
                     bool                 atEnd)
                : m_pGrid(pGrid),
                  m_current(kStartIter),
                  m_done(atEnd)
            {
                if (!m_done)
                {
                    Advance();
                }
            }

            ~Iterator()
            {
            }

            bool operator!=(const Iterator &other) const
            {
                (void)other;
                return !m_done;
            }

            int32_t operator*() const
            {
                return m_current;
            }

            Iterator &operator++()
            {
                Advance();
                return *this;
            }

        private:
            void Advance()
            {
                if (m_pGrid->m_usingSparseIndices)
                {
                    m_current = (m_current == kStartIter) ? m_pGrid->m_firstLiveIndex : m_pGrid->m_nextLiveIndex[m_current];
                    if (m_current < 0)
                    {
                        m_done = true;
                    }
                    return;
                }
                m_current = (m_current == kStartIter) ? 0 : (m_current + 1);
                while (m_current < m_pGrid->m_length)
                {
                    if (m_pGrid->m_agents[m_current] > TCount{})
                    {
                        return;
                    }
                    m_current++;
                }
                m_done = true;
            }

            PopulationGridBase *m_pGrid;
            int32_t m_current;
            bool m_done;
        };

        Iterator begin()
        {
            return Iterator(this, false);
        }

        Iterator end()
        {
            return Iterator(this, true);
        }

    private:
        // Adds i to the pending-delta list if it isn't already on it.
        void LinkDeltaIndex(int32_t i)
        {
            if (m_nextLiveDelta[i] != kUnlinked)
            {
                return;
            }
            m_nextLiveDelta[i] = (m_firstLiveDelta == kUnlinked) ? kLinkedTo : m_firstLiveDelta;
            m_firstLiveDelta = i;
        }

        // Links i to the front of the occupied-index list without changing
        // the occupied count (SetupSparseIndices relinks cells already
        // counted).
        void LinkPopIndex(int32_t i)
        {
            if (m_firstLiveIndex != kUnlinked)
            {
                m_prevLiveIndex[m_firstLiveIndex] = i;
                m_nextLiveIndex[i] = m_firstLiveIndex;
            }
            m_firstLiveIndex = i;
        }

        // LinkPopIndex plus counting the newly occupied cell.
        void AddPopIndex(int32_t i)
        {
            m_occupiedArea++;
            LinkPopIndex(i);
        }

        void RemovePopIndex(int32_t i)
        {
            m_occupiedArea--;
            if (m_firstLiveIndex == i)
            {
                m_firstLiveIndex = m_nextLiveIndex[i];
            }
            if (m_prevLiveIndex[i] >= 0)
            {
                m_nextLiveIndex[m_prevLiveIndex[i]] = m_nextLiveIndex[i];
            }
            if (m_nextLiveIndex[i] >= 0)
            {
                m_prevLiveIndex[m_nextLiveIndex[i]] = m_prevLiveIndex[i];
            }
            m_prevLiveIndex[i] = kUnlinked;
            m_nextLiveIndex[i] = kUnlinked;
        }

        static constexpr int32_t kUnlinked  = -1;
        static constexpr int32_t kLinkedTo  = -2;
        static constexpr int32_t kStartIter = -3;

        std::vector<TCount> m_agents;
        std::vector<TCount> m_deltas;
        int32_t m_occupiedArea;
        int64_t m_pop;
        int32_t m_updateCt;
        int32_t m_length;

        bool m_usingSparseIndices;
        int32_t m_firstLiveIndex;
        std::vector<int32_t> m_nextLiveIndex;
        std::vector<int32_t> m_prevLiveIndex;
        std::vector<int32_t> m_nextLiveDelta;
        int32_t m_firstLiveDelta;
    };
}
