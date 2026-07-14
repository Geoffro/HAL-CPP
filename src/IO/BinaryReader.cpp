#include "Hal/IO/BinaryReader.h"

#include "Hal/Log/Logger.h"

namespace Hal
{
    BinaryReader::BinaryReader(const std::string &path)
        : m_stream(path, std::ios::binary)
    {
        if (!m_stream.is_open())
        {
            Logger::Instance().Log(LogSeverity::Error, "BinaryReader failed to open file for reading: " + path);
        }
    }

    BinaryReader::~BinaryReader()
    {
    }

    bool BinaryReader::IsOpen() const
    {
        return m_stream.is_open();
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
