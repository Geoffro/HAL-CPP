// Ported from LEARN_HERE/Other/FileIOTest: a stress test that opens many
// output files at once, writes to each, and closes them. Java uses HAL's
// FileIO + Util.MakeDirs; the C++ port uses the standard library
// (<fstream> / <filesystem>) directly, which covers the same ground.
//
// Headless: creates fileIOtest/ with the files and prints a summary.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

int main()
{
    const int32_t nFiles = 200; // Java opens 1000; 200 keeps the demo tidy.
    std::filesystem::create_directories("fileIOtest");

    std::vector<std::ofstream> outs;
    outs.reserve(nFiles);
    for (int32_t i = 0; i < nFiles; i++)
    {
        outs.emplace_back("fileIOtest/test" + std::to_string(i) + ".csv");
    }
    for (std::ofstream &o : outs)
    {
        o << "test";
        o.close();
    }

    std::printf("wrote %d files into fileIOtest/\n", nFiles);
    return 0;
}
