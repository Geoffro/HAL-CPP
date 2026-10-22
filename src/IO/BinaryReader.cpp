#include "Hal/IO/BinaryReader.h"

#include "Hal/Log/Logger.h"

namespace Hal
{
    BinaryReader::BinaryReader(const std::string &path)
        : m_stream(path, std::ios::binary),
          m_path(path),
          m_loggedFailure(false)
    {
        if (!m_stream.is_open())
        {
            Logger::Instance().Log(LogSeverity::Error, "BinaryReader failed to open file for reading: " + path);
            m_loggedFailure = true;
        }
    }

    BinaryReader::~BinaryReader()
    {
    }

    bool BinaryReader::IsOpen() const
    {
        return m_stream.is_open();
    }

    bool BinaryReader::IsGood() const
    {
        return m_stream.is_open() && !m_stream.fail();
    }

    void BinaryReader::CheckStream()
    {
        if (m_stream.fail() && !m_loggedFailure)
        {
            Logger::Instance().Log(LogSeverity::Error, "BinaryReader read failed or ran past the end of: " + m_path);
            m_loggedFailure = true;
        }
    }

    bool BinaryReader::ReadBool()
    {
        bool value = false;
        ReadArray(&value, 1);
        return value;
    }

    int32_t BinaryReader::ReadInt32()
    {
        int32_t value = 0;
        ReadArray(&value, 1);
        return value;
    }

    int64_t BinaryReader::ReadInt64()
    {
        int64_t value = 0;
        ReadArray(&value, 1);
        return value;
    }

    uint32_t BinaryReader::ReadUint32()
    {
        uint32_t value = 0;
        ReadArray(&value, 1);
        return value;
    }

    uint64_t BinaryReader::ReadUint64()
    {
        uint64_t value = 0;
        ReadArray(&value, 1);
        return value;
    }

    double BinaryReader::ReadDouble()
    {
        double value = 0.0;
        ReadArray(&value, 1);
        return value;
    }
}
