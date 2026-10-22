#include "TestSupport.h"

#include <filesystem>

namespace TestHelpers
{
    std::string TempFilePath(const std::string &name)
    {
        return (std::filesystem::temp_directory_path() / name).string();
    }
}
