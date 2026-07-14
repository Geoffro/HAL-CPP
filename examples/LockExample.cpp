// Ported from LEARN_HERE/Other/LockExample: several threads writing to one
// file concurrently, serialized by a lock so their multi-line messages
// don't interleave. Java uses Util.MultiThread + a ReentrantLock; the C++
// port uses Hal::MultiThread + std::mutex.
//
// Headless: writes LockExample.txt and prints a summary.

#include "Hal/Concurrency/MultiThread.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>

static void ConcurrentWrite(std::ofstream     &out,
                            std::mutex        &lock,
                            bool               isLocking,
                            const std::string &text)
{
    if (isLocking)
    {
        lock.lock();
    }
    // Three writes that must stay contiguous per thread.
    out << text;
    out << text;
    out << text;
    if (isLocking)
    {
        lock.unlock();
    }
}

int main()
{
    std::ofstream out("LockExample.txt");
    std::mutex    lock;

    Hal::MultiThread(10, [&](int32_t i) {
        ConcurrentWrite(out, lock, true,
                        "Thread " + std::to_string(i) + " wrote this test bit of text!\n");
    });
    out.close();

    std::printf("10 threads wrote LockExample.txt under a lock\n");
    return 0;
}
