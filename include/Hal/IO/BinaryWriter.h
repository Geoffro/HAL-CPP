#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace Hal
{
    // Minimal binary serialization writer. Writes raw native-endian bytes
    // with no format header or byte-order translation, so files are only
    // meant to be read back on the same platform. A failed write logs an
    // error once and makes IsGood() return false.
    class BinaryWriter
    {
    public:
        explicit BinaryWriter(const std::string &path);
        ~BinaryWriter();

        BinaryWriter(const BinaryWriter &other) = delete;
        BinaryWriter &operator=(const BinaryWriter &other) = delete;

        bool IsOpen() const;

        // False once any write has failed.
        bool IsGood() const;

        void WriteBool(bool value);
        void WriteInt32(int32_t value);
        void WriteInt64(int64_t value);
        void WriteUint32(uint32_t value);
        void WriteUint64(uint64_t value);
        void WriteDouble(double value);

        // Writes count trivially-copyable elements of pData verbatim.
        template <typename T>
        void WriteArray(const T *pData,
                        int32_t  count)
        {
            static_assert(std::is_trivially_copyable<T>::value, "WriteArray requires a trivially-copyable element type");
            m_stream.write(reinterpret_cast<const char *>(pData), static_cast<std::streamsize>(sizeof(T)) * count);
            CheckStream();
        }

        template <typename T>
        void WriteArray(const std::vector<T> &data)
        {
            WriteArray(data.data(), static_cast<int32_t>(data.size()));
        }

    private:
        void CheckStream();

        std::ofstream m_stream;
        std::string   m_path;
        bool          m_loggedFailure;
    };
}
