#pragma once

#include <string>

namespace TestHelpers
{
    // Path for a scratch file named name in the OS temp directory.
    std::string TempFilePath(const std::string &name);
}
