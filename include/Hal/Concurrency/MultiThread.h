#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace Hal
{
    // Runs runFn(runIndex) for runIndex in [0, numRuns) on up to numThreads
    // threads, for independent replicate/parameter-sweep runs. runFn is
    // called concurrently, so shared mutable state is the caller's
    // responsibility. Runs are pulled from a shared counter, so uneven run
    // times still balance; the calling thread does work too, so
    // numThreads == 1 spawns nothing. If any run throws, the remaining runs
    // still execute and the first exception is rethrown afterward.
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
