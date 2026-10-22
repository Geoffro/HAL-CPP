#include "Hal/IO/BinaryWriter.h"

#include "Hal/Log/Logger.h"

namespace Hal
{
    BinaryWriter::BinaryWriter(const std::string &path)
        : m_stream(path, std::ios::binary | std::ios::trunc),
          m_path(path),
          m_loggedFailure(false)
    {
        if (!m_stream.is_open())
        {
            Logger::Instance().Log(LogSeverity::Error, "BinaryWriter failed to open file for writing: " + path);
            m_loggedFailure = true;
        }
    }

    BinaryWriter::~BinaryWriter()
    {
    }

    bool BinaryWriter::IsOpen() const
    {
        return m_stream.is_open();
    }

    bool BinaryWriter::IsGood() const
    {
        return m_stream.is_open() && !m_stream.fail();
    }

    void BinaryWriter::CheckStream()
    {
        if (m_stream.fail() && !m_loggedFailure)
        {
            Logger::Instance().Log(LogSeverity::Error, "BinaryWriter write failed: " + m_path);
            m_loggedFailure = true;
        }
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
