#pragma once

#include <cstdint>
#include <mutex>
#include <utility>

namespace Hal
{
    // Single-producer/single-consumer triple buffer, the concrete
    // mechanism behind docs/DESIGN.md's sim/render threading contract:
    // the simulation thread writes state into a private back buffer and
    // Publish()es it; the render loop Acquire()s the most recently
    // published buffer and reads a consistent, tear-free snapshot. Unlike
    // Java HAL (which mutates Swing components straight from the sim
    // thread, unsafe per Swing's own contract), producer and consumer here
    // never touch the same T instance concurrently.
    //
    // Three slots so the producer always has a slot to write and the
    // consumer always has a slot to read, with a third holding the latest
    // completed frame in between — the producer never blocks on the
    // consumer and vice versa. "Latest wins": if the producer publishes
    // several frames between two Acquire() calls, the consumer skips
    // straight to the newest, exactly what a render loop wants (drop stale
    // frames rather than fall behind).
    //
    // T must be default-constructible (or copy-constructible for the
    // initial-value constructor). The producer must not retain the
    // reference from BackBuffer() across a Publish(), nor the consumer the
    // reference from FrontBuffer() across an Acquire() — those swaps
    // repoint which slot each refers to.
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
