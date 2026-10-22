#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace Hal
{
    // Reads the raw native-endian format written by BinaryWriter. A failed
    // or short read logs an error once and makes IsGood() return false.
    class BinaryReader
    {
    public:
        explicit BinaryReader(const std::string &path);
        ~BinaryReader();

        BinaryReader(const BinaryReader &other) = delete;
        BinaryReader &operator=(const BinaryReader &other) = delete;

        bool IsOpen() const;

        // False once any read has failed (including reading past the end).
        bool IsGood() const;

        bool ReadBool();
        int32_t ReadInt32();
        int64_t ReadInt64();
        uint32_t ReadUint32();
        uint64_t ReadUint64();
        double ReadDouble();

        // Reads count trivially-copyable elements into pData verbatim.
        template <typename T>
        void ReadArray(T       *pData,
                       int32_t  count)
        {
            static_assert(std::is_trivially_copyable<T>::value, "ReadArray requires a trivially-copyable element type");
            m_stream.read(reinterpret_cast<char *>(pData), static_cast<std::streamsize>(sizeof(T)) * count);
            CheckStream();
        }

        template <typename T>
        void ReadArray(std::vector<T> &data)
        {
            ReadArray(data.data(), static_cast<int32_t>(data.size()));
        }

    private:
        void CheckStream();

        std::ifstream m_stream;
        std::string   m_path;
        bool          m_loggedFailure;
    };
}
