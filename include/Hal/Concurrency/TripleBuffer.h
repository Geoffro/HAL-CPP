#pragma once

#include <cstdint>
#include <mutex>
#include <utility>

namespace Hal
{
    // Single-producer/single-consumer triple buffer for handing simulation
    // state to a render loop. The producer writes BackBuffer() and
    // Publish()es it; the consumer Acquire()s the newest published frame
    // and reads FrontBuffer(). Neither side blocks the other or touches the
    // same slot concurrently, and frames published between two Acquire()
    // calls are skipped in favor of the newest. Publish() hands the back
    // slot off, so the producer gets an older frame's slot to rewrite next.
    // References from BackBuffer()/FrontBuffer() are invalidated by
    // Publish()/Acquire() respectively.
    template <typename T>
    class TripleBuffer
    {
    public:
        TripleBuffer()
            : m_buffers{},
              m_backIndex(0),
              m_readyIndex(1),
              m_frontIndex(2),
              m_hasNewFrame(false),
              m_mutex()
        {
        }

        explicit TripleBuffer(const T &initial)
            : m_buffers{initial, initial, initial},
              m_backIndex(0),
              m_readyIndex(1),
              m_frontIndex(2),
              m_hasNewFrame(false),
              m_mutex()
        {
        }

        ~TripleBuffer()
        {
        }

        TripleBuffer(const TripleBuffer &other) = delete;
        TripleBuffer &operator=(const TripleBuffer &other) = delete;

        // Producer side: the slot to write the next frame into. Stable
        // until the next Publish().
        T &BackBuffer()
        {
            return m_buffers[m_backIndex];
        }

        // Producer side: hands the just-written back buffer off as the
        // newest ready frame, taking the old ready slot to write next.
        void Publish()
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            std::swap(m_backIndex, m_readyIndex);
            m_hasNewFrame = true;
        }

        // Consumer side: if a newer frame has been published since the
        // last Acquire(), swaps it into the front slot and returns true;
        // otherwise leaves the front slot unchanged and returns false.
        bool Acquire()
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_hasNewFrame)
            {
                return false;
            }
            std::swap(m_frontIndex, m_readyIndex);
            m_hasNewFrame = false;
            return true;
        }

        // Consumer side: the most recently Acquire()d frame. Stable until
        // the next Acquire().
        const T &FrontBuffer() const
        {
            return m_buffers[m_frontIndex];
        }

    private:
        T m_buffers[3];
        int32_t m_backIndex;
        int32_t m_readyIndex;
        int32_t m_frontIndex;
        bool m_hasNewFrame;
        mutable std::mutex m_mutex;
    };
}
