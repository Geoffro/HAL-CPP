#pragma once

#include <cstdint>
#include <string>

namespace Hal
{
    enum class LogSeverity : uint32_t
    {
        Debug,
        Info,
        Warning,
        Error
    };

    class Logger
    {
    public:
        Logger();
        ~Logger();

        static Logger &Instance();

        void Log(LogSeverity        severity,
                 const std::string &message);

        void SetMinSeverity(LogSeverity severity);

    private:
        LogSeverity m_minSeverity;
    };

    const char *ToString(LogSeverity severity);
}
