#include <doctest/doctest.h>

#include "Hal/Concurrency/MultiThread.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

TEST_CASE("Hal::MultiThread runs every index exactly once, writing distinct result slots")
{
    const int32_t numRuns = 1000;
    std::vector<int32_t> results(static_cast<size_t>(numRuns), -1);

    Hal::MultiThread(numRuns, 4, [&](int32_t runIndex)
    {
        results[static_cast<size_t>(runIndex)] = runIndex * runIndex;
    });

    for (int32_t i = 0; i < numRuns; i++)
    {
        CAPTURE(i);
        CHECK(results[static_cast<size_t>(i)] == i * i);
    }
}

TEST_CASE("Hal::MultiThread never exceeds the requested concurrency and actually parallelizes")
{
    const int32_t numRuns = 200;
    const int32_t numThreads = 4;
    std::atomic<int32_t> active(0);
    std::atomic<int32_t> maxActive(0);

    Hal::MultiThread(numRuns, numThreads, [&](int32_t /*runIndex*/)
    {
        int32_t nowActive = active.fetch_add(1) + 1;
        int32_t observedMax = maxActive.load();
        while (nowActive > observedMax && !maxActive.compare_exchange_weak(observedMax, nowActive))
        {
            // observedMax refreshed by compare_exchange_weak on failure.
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        active.fetch_sub(1);
    });

    CHECK(maxActive.load() <= numThreads);
    // With a 1ms hold per run and 200 runs, several should overlap.
    CHECK(maxActive.load() >= 2);
}

TEST_CASE("Hal::MultiThread with a single thread runs serially (no overlap)")
{
    const int32_t numRuns = 50;
    std::atomic<int32_t> active(0);
    std::atomic<int32_t> maxActive(0);

    Hal::MultiThread(numRuns, 1, [&](int32_t /*runIndex*/)
    {
        int32_t nowActive = active.fetch_add(1) + 1;
        int32_t observedMax = maxActive.load();
        while (nowActive > observedMax && !maxActive.compare_exchange_weak(observedMax, nowActive))
        {
        }
        active.fetch_sub(1);
    });

    CHECK(maxActive.load() == 1);
}

TEST_CASE("Hal::MultiThread propagates an exception thrown by a run")
{
    const int32_t numRuns = 20;
    std::atomic<int32_t> completed(0);

    CHECK_THROWS_AS(
        Hal::MultiThread(numRuns, 4, [&](int32_t runIndex)
        {
            if (runIndex == 7)
            {
                throw std::runtime_error("run 7 failed");
            }
            completed.fetch_add(1);
        }),
        std::runtime_error);

    // The other independent runs still execute despite one throwing.
    CHECK(completed.load() == numRuns - 1);
}

TEST_CASE("Hal::MultiThread handles zero and negative run counts as no-ops")
{
    std::atomic<int32_t> calls(0);
    Hal::MultiThread(0, 4, [&](int32_t /*runIndex*/)
    {
        calls.fetch_add(1);
    });
    Hal::MultiThread(-5, 4, [&](int32_t /*runIndex*/)
    {
        calls.fetch_add(1);
    });
    CHECK(calls.load() == 0);
}

TEST_CASE("Hal::MultiThread hardware-concurrency overload runs every index once")
{
    const int32_t numRuns = 300;
    std::vector<int32_t> results(static_cast<size_t>(numRuns), 0);

    Hal::MultiThread(numRuns, [&](int32_t runIndex)
    {
        results[static_cast<size_t>(runIndex)] = runIndex + 1;
    });

    for (int32_t i = 0; i < numRuns; i++)
    {
        CAPTURE(i);
        CHECK(results[static_cast<size_t>(i)] == i + 1);
    }
}
