#include <doctest/doctest.h>

#include "Hal/Concurrency/TripleBuffer.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

TEST_CASE("Hal::TripleBuffer: Acquire returns false until the first Publish, then exposes the published frame")
{
    Hal::TripleBuffer<int32_t> buffer;

    // Nothing published yet.
    CHECK_FALSE(buffer.Acquire());

    buffer.BackBuffer() = 42;
    buffer.Publish();

    CHECK(buffer.Acquire());
    CHECK(buffer.FrontBuffer() == 42);

    // No new frame since the acquire.
    CHECK_FALSE(buffer.Acquire());
    CHECK(buffer.FrontBuffer() == 42);
}

TEST_CASE("Hal::TripleBuffer: a single Acquire after several Publishes yields the newest frame (latest wins)")
{
    Hal::TripleBuffer<int32_t> buffer;

    buffer.BackBuffer() = 1;
    buffer.Publish();
    buffer.BackBuffer() = 2;
    buffer.Publish();
    buffer.BackBuffer() = 3;
    buffer.Publish();

    CHECK(buffer.Acquire());
    CHECK(buffer.FrontBuffer() == 3);
    CHECK_FALSE(buffer.Acquire());
}

TEST_CASE("Hal::TripleBuffer: the back buffer the producer writes is never the front buffer the consumer reads")
{
    // Publishing must not hand the consumer a slot the producer is still
    // writing into: after publish+acquire, the next BackBuffer() must be a
    // different object than the current FrontBuffer().
    Hal::TripleBuffer<int32_t> buffer;

    buffer.BackBuffer() = 10;
    buffer.Publish();
    REQUIRE(buffer.Acquire());

    const int32_t *pFront = &buffer.FrontBuffer();
    int32_t *pBack = &buffer.BackBuffer();
    CHECK(pFront != pBack);
}

TEST_CASE("Hal::TripleBuffer: initial-value constructor seeds every slot so an early Acquire reads it")
{
    Hal::TripleBuffer<int32_t> buffer(7);
    CHECK(buffer.FrontBuffer() == 7);
}

// The core guarantee: with a producer thread hammering Publish and a
// consumer thread hammering Acquire, the consumer must never observe a
// half-written (torn) frame, and the frames it sees must arrive in the
// order they were produced. Each published frame fills the whole vector
// with one monotonically-increasing value, so "all elements equal" ==
// "not torn" and the shared value == "which frame".
TEST_CASE("Hal::TripleBuffer: concurrent producer/consumer never tears a frame and preserves publish order")
{
    const int32_t frameLength = 4096;
    const int32_t frameCount = 200000;

    Hal::TripleBuffer<std::vector<int32_t>> buffer(std::vector<int32_t>(frameLength, 0));
    std::atomic<bool> producerDone(false);

    std::thread producer([&]()
    {
        for (int32_t frame = 1; frame <= frameCount; frame++)
        {
            std::vector<int32_t> &back = buffer.BackBuffer();
            for (int32_t i = 0; i < frameLength; i++)
            {
                back[i] = frame;
            }
            buffer.Publish();
        }
        producerDone.store(true);
    });

    int32_t tornFrames = 0;
    int32_t orderViolations = 0;
    int32_t lastSeen = 0;
    int32_t acquiredCount = 0;

    // Read producerDone before Acquire each pass: producerDone is stored
    // strictly after the producer's final Publish, so once we observe it
    // true, the following Acquire is guaranteed to see any last frame.
    while (true)
    {
        bool done = producerDone.load();
        if (buffer.Acquire())
        {
            const std::vector<int32_t> &front = buffer.FrontBuffer();
            int32_t value = front[0];
            for (int32_t i = 1; i < frameLength; i++)
            {
                if (front[i] != value)
                {
                    tornFrames++;
                    break;
                }
            }
            if (value < lastSeen)
            {
                orderViolations++;
            }
            lastSeen = value;
            acquiredCount++;
        }
        else if (done)
        {
            break;
        }
    }

    producer.join();

    CHECK(tornFrames == 0);
    CHECK(orderViolations == 0);
    // The consumer should have seen at least the final frame; "latest
    // wins" means it may skip many, so we only assert it made progress.
    CHECK(acquiredCount > 0);
}
