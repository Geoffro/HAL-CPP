#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>

#include "Hal/Log/Logger.h"

#include <cstring>

TEST_CASE("LogSeverity ToString returns the expected label for each level")
{
    CHECK(std::strcmp(Hal::ToString(Hal::LogSeverity::Debug), "DEBUG") == 0);
    CHECK(std::strcmp(Hal::ToString(Hal::LogSeverity::Info), "INFO") == 0);
    CHECK(std::strcmp(Hal::ToString(Hal::LogSeverity::Warning), "WARNING") == 0);
    CHECK(std::strcmp(Hal::ToString(Hal::LogSeverity::Error), "ERROR") == 0);
}

TEST_CASE("Logger singleton is reachable and accepts messages at every severity")
{
    Hal::Logger &logger = Hal::Logger::Instance();

    logger.SetMinSeverity(Hal::LogSeverity::Debug);
    logger.Log(Hal::LogSeverity::Debug, "debug message");
    logger.Log(Hal::LogSeverity::Info, "info message");
    logger.Log(Hal::LogSeverity::Warning, "warning message");
    logger.Log(Hal::LogSeverity::Error, "error message");
}
