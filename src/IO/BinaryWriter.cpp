#include "Hal/IO/BinaryWriter.h"

#include "Hal/Log/Logger.h"

namespace Hal
{
    BinaryWriter::BinaryWriter(const std::string &path)
        : m_stream(path, std::ios::binary | std::ios::trunc)
    {
        if (!m_stream.is_open())
        {
            Logger::Instance().Log(LogSeverity::Error, "BinaryWriter failed to open file for writing: " + path);
        }
    }

    BinaryWriter::~BinaryWriter()
    {
    }

    bool BinaryWriter::IsOpen() const
    {
        return m_stream.is_open();
    }

    void BinaryWriter::WriteBool(bool value)
    {
        WriteArray(&value, 1);
    }

    void BinaryWriter::WriteInt32(int32_t value)
    {
        WriteArray(&value, 1);
    }

    void BinaryWriter::WriteInt64(int64_t value)
    {
        WriteArray(&value, 1);
    }

    void BinaryWriter::WriteUint32(uint32_t value)
    {
        WriteArray(&value, 1);
    }

    void BinaryWriter::WriteUint64(uint64_t value)
    {
        WriteArray(&value, 1);
    }

    void BinaryWriter::WriteDouble(double value)
    {
        WriteArray(&value, 1);
    }
}
