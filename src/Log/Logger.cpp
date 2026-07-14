#include "Hal/Log/Logger.h"

#include <iostream>

namespace Hal
{
    Logger::Logger()
        : m_minSeverity(LogSeverity::Info)
    {
    }

    Logger::~Logger()
    {
    }

    Logger &Logger::Instance()
    {
        static Logger instance;
        return instance;
    }

    void Logger::SetMinSeverity(LogSeverity severity)
    {
        m_minSeverity = severity;
    }

    void Logger::Log(LogSeverity        severity,
                     const std::string &message)
    {
        if (severity < m_minSeverity)
        {
            return;
        }

        std::ostream &stream = (severity == LogSeverity::Error) ? std::cerr : std::cout;
        stream << "[" << ToString(severity) << "] " << message << std::endl;
    }

    const char *ToString(LogSeverity severity)
    {
        switch (severity)
        {
            case LogSeverity::Debug:
            {
                return "DEBUG";
            }
            case LogSeverity::Info:
            {
                return "INFO";
            }
            case LogSeverity::Warning:
            {
                return "WARNING";
            }
            case LogSeverity::Error:
            {
                return "ERROR";
            }
        }

        return "UNKNOWN";
    }
}
