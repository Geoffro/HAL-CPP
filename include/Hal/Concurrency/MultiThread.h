#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace Hal
{
    // Runs runFn(runIndex) for runIndex in [0, numRuns), with at most
    // numThreads running concurrently — ported from Util.MultiThread. The
    // intended use is coarse replicate/parameter-sweep parallelism: each
    // run is independent and typically writes its result into its own slot
    // of a shared array (index runIndex), so no synchronization between
    // runs is needed. runFn is a template callable (not a std::function or
    // named-interface object, per docs/DESIGN.md's callback-style rule),
    // called concurrently from several threads, so it must be safe to
    // invoke that way (independent per-run writes are; shared mutable
    // state is the caller's responsibility, exactly as in the Java
    // original).
    //
    // Work is pulled from a shared atomic counter rather than pre-assigned,
    // so uneven run durations still balance across threads. numThreads-1
    // worker threads are spawned and the calling thread runs work too, so
    // numThreads == 1 executes serially on the caller with no threads
    // spawned.
    //
    // Improvement over the Java version, whose ExecutorService silently
    // drops exceptions thrown by a run: if any run throws, the first such
    // exception is re-thrown from MultiThread after all runs finish (the
    // remaining runs still execute — they are independent).
    template <typename ParallelFn>
    void MultiThread(int32_t    numRuns,
                     int32_t    numThreads,
                     ParallelFn runFn)
    {
        if (numRuns <= 0)
        {
            return;
        }
        int32_t workerCount = (numThreads < numRuns) ? numThreads : numRuns;
        if (workerCount < 1)
        {
            workerCount = 1;
        }

        std::atomic<int32_t> nextIndex(0);
        std::mutex exceptionMutex;
        std::exception_ptr firstException = nullptr;

        auto worker = [&]()
        {
            while (true)
            {
                int32_t runIndex = nextIndex.fetch_add(1);
                if (runIndex >= numRuns)
                {
                    break;
                }
                try
                {
                    runFn(runIndex);
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(exceptionMutex);
                    if (firstException == nullptr)
                    {
                        firstException = std::current_exception();
                    }
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(workerCount - 1));
        for (int32_t i = 0; i < (workerCount - 1); i++)
        {
            threads.emplace_back(worker);
        }
        worker();
        for (std::thread &thread : threads)
        {
            thread.join();
        }

        if (firstException != nullptr)
        {
            std::rethrow_exception(firstException);
        }
    }

    // As above, defaulting the concurrency to the hardware's reported
    // thread count (falling back to 1 if it can't be determined).
    template <typename ParallelFn>
    void MultiThread(int32_t    numRuns,
                     ParallelFn runFn)
    {
        int32_t hardwareThreads = static_cast<int32_t>(std::thread::hardware_concurrency());
        if (hardwareThreads < 1)
        {
            hardwareThreads = 1;
        }
        MultiThread(numRuns, hardwareThreads, runFn);
    }
}
