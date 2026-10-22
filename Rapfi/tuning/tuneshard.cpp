/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "tuneshard.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

namespace Tuning {
namespace {

    constexpr std::array<uint8_t, 8> Magic            = {'R', 'F', 'T', 'U', 'N', 'E', '0', '1'};
    constexpr uint32_t               FormatVersion    = 11;
    constexpr size_t                 SectionCount     = 10;
    constexpr size_t                 FingerprintBytes = 32;

    enum Section : size_t {
        BoardSizes,
        Results,
        StaticEvals,
        BestCandidates,
        EvalOffsets,
        EvalTerms,
        PolicyOffsets,
        PolicyCandidates,
        PolicyTargetOffsets,
        PolicyTargets,
    };

    struct SectionMeta
    {
        uint64_t byteSize;
        uint32_t crc32;
    };

    class Crc32
    {
    public:
        void addByte(uint8_t byte) { state = (state >> 8) ^ table()[(state ^ byte) & 0xffU]; }

        void addBytes(const void *data, size_t size)
        {
            const uint8_t *bytes = static_cast<const uint8_t *>(data);
            for (size_t i = 0; i < size; i++)
                addByte(bytes[i]);
        }

        template <typename UInt>
        void addLittleEndian(UInt value)
        {
            static_assert(std::is_unsigned<UInt>::value, "CRC input must be unsigned");
            for (size_t i = 0; i < sizeof(UInt); i++) {
                addByte(static_cast<uint8_t>(value));
                value >>= 8;
            }
        }

        uint32_t value() const { return state ^ 0xffffffffU; }

    private:
        static const std::array<uint32_t, 256> &table()
        {
            static const std::array<uint32_t, 256> values = [] {
                std::array<uint32_t, 256> result {};
                for (uint32_t i = 0; i < result.size(); i++) {
                    uint32_t value = i;
                    for (int bit = 0; bit < 8; bit++)
                        value = (value >> 1) ^ (0xedb88320U & (0U - (value & 1U)));
                    result[i] = value;
                }
                return result;
            }();
            return values;
        }

        uint32_t state = 0xffffffffU;
    };

    uint64_t checkedByteSize(size_t count, size_t elementSize, const char *section)
    {
        if (count > std::numeric_limits<uint64_t>::max() / elementSize)
            throw std::length_error(std::string("prepared shard ") + section + " size overflows");
        return uint64_t(count) * elementSize;
    }

    std::array<uint8_t, FingerprintBytes> decodeFingerprint(const std::string &fingerprint)
    {
        std::array<uint8_t, FingerprintBytes> bytes {};
        if (fingerprint.empty())
            return bytes;
        if (fingerprint.size() != bytes.size() * 2)
            throw std::invalid_argument("prepared shard fingerprint must contain 64 hex digits");

        auto hexValue = [](char digit) -> int {
            if (digit >= '0' && digit <= '9')
                return digit - '0';
            if (digit >= 'a' && digit <= 'f')
                return digit - 'a' + 10;
            if (digit >= 'A' && digit <= 'F')
                return digit - 'A' + 10;
            return -1;
        };
        for (size_t i = 0; i < bytes.size(); i++) {
            int high = hexValue(fingerprint[i * 2]);
            int low  = hexValue(fingerprint[i * 2 + 1]);
            if (high < 0 || low < 0)
                throw std::invalid_argument("prepared shard fingerprint contains non-hex digits");
            bytes[i] = static_cast<uint8_t>((high << 4) | low);
        }
        return bytes;
    }

    class BufferedWriter
    {
    public:
        explicit BufferedWriter(std::ostream &out) : out(out) {}

        void byte(uint8_t value)
        {
            if (used == buffer.size())
                flush();
            buffer[used++] = static_cast<char>(value);
        }

        template <typename UInt>
        void littleEndian(UInt value)
        {
            static_assert(std::is_unsigned<UInt>::value, "serialized integer must be unsigned");
            for (size_t i = 0; i < sizeof(UInt); i++) {
                byte(static_cast<uint8_t>(value));
                value >>= 8;
            }
        }

        void flush()
        {
            if (used != 0) {
                out.write(buffer.data(), used);
                used = 0;
            }
        }

    private:
        std::ostream           &out;
        std::array<char, 65536> buffer;
        size_t                  used = 0;
    };

    template <typename UInt>
    void writeLittleEndian(BufferedWriter &out, UInt value)
    {
        out.littleEndian(value);
    }

    uint8_t readByte(std::istream &in)
    {
        char value;
        if (!in.get(value))
            throw std::runtime_error("prepared shard is truncated");
        return static_cast<uint8_t>(value);
    }

    template <typename UInt>
    UInt readLittleEndian(std::istream &in)
    {
        static_assert(std::is_unsigned<UInt>::value, "serialized integer must be unsigned");
        UInt value = 0;
        for (size_t i = 0; i < sizeof(UInt); i++)
            value |= UInt(readByte(in)) << (i * 8);
        return value;
    }

    int16_t signed16(uint16_t bits)
    {
        return bits <= uint16_t(std::numeric_limits<int16_t>::max())
                   ? static_cast<int16_t>(bits)
                   : static_cast<int16_t>(int32_t(bits) - 0x10000);
    }

    bool isLittleEndianHost()
    {
        const uint16_t value = 1;
        return *reinterpret_cast<const uint8_t *>(&value) == 1;
    }

    template <typename Encode>
    uint32_t sectionCrc(Encode encode)
    {
        Crc32 crc;
        encode(crc);
        return crc.value();
    }

    class CheckedSectionReader
    {
    public:
        CheckedSectionReader(std::istream &in, const SectionMeta &meta, bool verifyChecksum)
            : in(in)
            , meta(meta)
            , verifyChecksum(verifyChecksum)
        {}

        uint8_t byte()
        {
            if (bufferPosition == bufferSize)
                refill();
            uint8_t value = static_cast<uint8_t>(buffer[bufferPosition++]);
            if (verifyChecksum)
                crc.addByte(value);
            bytesRead++;
            return value;
        }

        template <typename UInt>
        UInt littleEndian()
        {
            static_assert(std::is_unsigned<UInt>::value, "serialized integer must be unsigned");
            UInt value = 0;
            for (size_t i = 0; i < sizeof(UInt); i++)
                value |= UInt(byte()) << (i * 8);
            return value;
        }

        void raw(void *destination, size_t size)
        {
            if (bufferPosition != bufferSize || size > meta.byteSize - bytesRead)
                throw std::logic_error("invalid prepared shard raw section read");
            in.read(static_cast<char *>(destination), size);
            if (size_t(in.gcount()) != size)
                throw std::runtime_error("prepared shard is truncated");
            if (verifyChecksum)
                crc.addBytes(destination, size);
            bytesRead += size;
        }

        void finish(const char *name)
        {
            if (bytesRead != meta.byteSize)
                throw std::runtime_error(std::string("prepared shard ") + name
                                         + " section size mismatch");
            if (verifyChecksum && crc.value() != meta.crc32)
                throw std::runtime_error(std::string("prepared shard ") + name
                                         + " section checksum mismatch");
        }

    private:
        void refill()
        {
            uint64_t remaining = meta.byteSize - bytesRead;
            if (remaining == 0)
                throw std::runtime_error("prepared shard section is truncated");
            bufferSize = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
            in.read(buffer.data(), bufferSize);
            if (size_t(in.gcount()) != bufferSize)
                throw std::runtime_error("prepared shard is truncated");
            bufferPosition = 0;
        }

        std::istream           &in;
        const SectionMeta      &meta;
        Crc32                   crc;
        uint64_t                bytesRead = 0;
        std::array<char, 65536> buffer;
        size_t                  bufferPosition = 0;
        size_t                  bufferSize     = 0;
        bool                    verifyChecksum;
    };

    std::array<SectionMeta, SectionCount> makeSectionMetadata(const PreparedCorpus &corpus)
    {
        if (corpus.size() > std::numeric_limits<uint32_t>::max()
            || corpus.evalTerms().size() > std::numeric_limits<uint32_t>::max()
            || corpus.policyCandidates().size() > std::numeric_limits<uint32_t>::max()
            || corpus.policyTargets().size() > std::numeric_limits<uint32_t>::max())
            throw std::length_error("prepared shard count exceeds 32-bit format capacity");

        std::array<SectionMeta, SectionCount> metadata;
        metadata[BoardSizes].byteSize =
            checkedByteSize(corpus.boardSizes().size(), 1, "board size");
        metadata[BoardSizes].crc32 = sectionCrc([&](Crc32 &crc) {
            for (uint8_t value : corpus.boardSizes())
                crc.addByte(value);
        });
        metadata[Results].byteSize = checkedByteSize(corpus.results().size(), 1, "result");
        metadata[Results].crc32    = sectionCrc([&](Crc32 &crc) {
            for (uint8_t value : corpus.results())
                crc.addByte(value);
        });
        metadata[StaticEvals].byteSize =
            checkedByteSize(corpus.staticEvals().size(), 2, "static eval");
        metadata[StaticEvals].crc32 = sectionCrc([&](Crc32 &crc) {
            for (int16_t value : corpus.staticEvals())
                crc.addLittleEndian(static_cast<uint16_t>(value));
        });
        metadata[BestCandidates].byteSize =
            checkedByteSize(corpus.bestCandidates().size(), 2, "best candidate");
        metadata[BestCandidates].crc32 = sectionCrc([&](Crc32 &crc) {
            for (uint16_t value : corpus.bestCandidates())
                crc.addLittleEndian(value);
        });
        metadata[EvalOffsets].byteSize =
            checkedByteSize(corpus.evalOffsets().size(), 4, "value offset");
        metadata[EvalOffsets].crc32  = sectionCrc([&](Crc32 &crc) {
            for (uint32_t value : corpus.evalOffsets())
                crc.addLittleEndian(value);
        });
        metadata[EvalTerms].byteSize = checkedByteSize(corpus.evalTerms().size(), 4, "value term");
        metadata[EvalTerms].crc32    = sectionCrc([&](Crc32 &crc) {
            for (const TuneCoeff &term : corpus.evalTerms()) {
                crc.addLittleEndian(static_cast<uint16_t>(term.coeff));
                crc.addLittleEndian(term.index);
            }
        });
        metadata[PolicyOffsets].byteSize =
            checkedByteSize(corpus.policyOffsets().size(), 4, "policy offset");
        metadata[PolicyOffsets].crc32       = sectionCrc([&](Crc32 &crc) {
            for (uint32_t value : corpus.policyOffsets())
                crc.addLittleEndian(value);
        });
        metadata[PolicyCandidates].byteSize = checkedByteSize(corpus.policyCandidates().size(),
                                                              sizeof(PolicyCandidate),
                                                              "policy candidate");
        metadata[PolicyCandidates].crc32    = sectionCrc([&](Crc32 &crc) {
            for (const PolicyCandidate &candidate : corpus.policyCandidates())
                for (ParameterId index : candidate.indices)
                    crc.addLittleEndian(index);
        });
        metadata[PolicyTargetOffsets].byteSize =
            checkedByteSize(corpus.policyTargetOffsets().size(), 4, "policy target offset");
        metadata[PolicyTargetOffsets].crc32 = sectionCrc([&](Crc32 &crc) {
            for (uint32_t value : corpus.policyTargetOffsets())
                crc.addLittleEndian(value);
        });
        metadata[PolicyTargets].byteSize =
            checkedByteSize(corpus.policyTargets().size(), 4, "policy target");
        metadata[PolicyTargets].crc32 = sectionCrc([&](Crc32 &crc) {
            for (const PolicyTargetTerm &target : corpus.policyTargets()) {
                crc.addLittleEndian(target.candidate);
                crc.addLittleEndian(target.weight);
            }
        });
        return metadata;
    }

    void validateSectionSizes(const std::array<SectionMeta, SectionCount> &metadata,
                              uint32_t                                     sampleCount,
                              uint32_t                                     evalTermCount,
                              uint32_t                                     policyCandidateCount,
                              uint32_t                                     policyTargetCount)
    {
        std::array<uint64_t, SectionCount> expected = {
            checkedByteSize(sampleCount, 1, "board size"),
            checkedByteSize(sampleCount, 1, "result"),
            checkedByteSize(sampleCount, 2, "static eval"),
            checkedByteSize(sampleCount, 2, "best candidate"),
            checkedByteSize(uint64_t(sampleCount) + 1, 4, "value offset"),
            checkedByteSize(evalTermCount, 4, "value term"),
            checkedByteSize(uint64_t(sampleCount) + 1, 4, "policy offset"),
            checkedByteSize(policyCandidateCount, sizeof(PolicyCandidate), "policy candidate"),
            checkedByteSize(uint64_t(sampleCount) + 1, 4, "policy target offset"),
            checkedByteSize(policyTargetCount, 4, "policy target"),
        };
        for (size_t i = 0; i < SectionCount; i++)
            if (metadata[i].byteSize != expected[i])
                throw std::runtime_error("prepared shard header contains an invalid section size");
    }

}  // namespace

void writePreparedShard(const std::filesystem::path &path,
                        const PreparedCorpus        &corpus,
                        const std::string           &fingerprint,
                        uint64_t                     shardOrdinal)
{
    auto metadata         = makeSectionMetadata(corpus);
    auto fingerprintBytes = decodeFingerprint(fingerprint);
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::filesystem::path tempPath = path;
    tempPath += ".tmp";

    if (std::filesystem::exists(path) || std::filesystem::exists(tempPath))
        throw std::runtime_error("prepared shard target already exists: " + path.string());

    try {
        std::ofstream out(tempPath, std::ios::binary);
        if (!out)
            throw std::runtime_error("unable to create prepared shard: " + tempPath.string());
        BufferedWriter writer(out);

        for (uint8_t byte : Magic)
            writer.byte(byte);
        writeLittleEndian(writer, FormatVersion);
        uint32_t flags = (!corpus.evalTerms().empty() ? 1U : 0U)
                         | (!corpus.policyCandidates().empty() ? 2U : 0U)
                         | (!corpus.policyTargets().empty() ? 4U : 0U);
        writeLittleEndian(writer, flags);
        writeLittleEndian(writer, static_cast<uint32_t>(corpus.size()));
        writeLittleEndian(writer, static_cast<uint32_t>(corpus.evalTerms().size()));
        writeLittleEndian(writer, static_cast<uint32_t>(corpus.policyCandidates().size()));
        writeLittleEndian(writer, static_cast<uint32_t>(corpus.policyTargets().size()));
        writeLittleEndian(writer, shardOrdinal);
        for (uint8_t byte : fingerprintBytes)
            writer.byte(byte);
        for (const SectionMeta &section : metadata) {
            writeLittleEndian(writer, section.byteSize);
            writeLittleEndian(writer, section.crc32);
        }

        for (uint8_t value : corpus.boardSizes())
            writer.byte(value);
        for (uint8_t value : corpus.results())
            writer.byte(value);
        for (int16_t value : corpus.staticEvals())
            writeLittleEndian(writer, static_cast<uint16_t>(value));
        for (uint16_t value : corpus.bestCandidates())
            writeLittleEndian(writer, value);
        for (uint32_t value : corpus.evalOffsets())
            writeLittleEndian(writer, value);
        for (const TuneCoeff &term : corpus.evalTerms()) {
            writeLittleEndian(writer, static_cast<uint16_t>(term.coeff));
            writeLittleEndian(writer, term.index);
        }
        for (uint32_t value : corpus.policyOffsets())
            writeLittleEndian(writer, value);
        for (const PolicyCandidate &candidate : corpus.policyCandidates())
            for (ParameterId index : candidate.indices)
                writeLittleEndian(writer, index);
        for (uint32_t value : corpus.policyTargetOffsets())
            writeLittleEndian(writer, value);
        for (const PolicyTargetTerm &target : corpus.policyTargets()) {
            writeLittleEndian(writer, target.candidate);
            writeLittleEndian(writer, target.weight);
        }
        writer.flush();
        out.flush();
        if (!out)
            throw std::runtime_error("failed to write prepared shard: " + tempPath.string());
        out.close();
        if (!out)
            throw std::runtime_error("failed to close prepared shard: " + tempPath.string());

        std::error_code renameError;
        std::filesystem::rename(tempPath, path, renameError);
        if (renameError)
            throw std::runtime_error("failed to publish prepared shard: " + renameError.message());
    }
    catch (...) {
        std::error_code ignored;
        std::filesystem::remove(tempPath, ignored);
        throw;
    }
}

PreparedCorpus readPreparedShard(const std::filesystem::path &path,
                                 bool                         verifyChecksum,
                                 size_t                       maxAllocationBytes,
                                 const std::string           &expectedFingerprint,
                                 uint64_t                     expectedShardOrdinal)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("unable to open prepared shard: " + path.string());

    for (uint8_t expected : Magic)
        if (readByte(in) != expected)
            throw std::runtime_error("prepared shard magic is invalid");
    uint32_t version = readLittleEndian<uint32_t>(in);
    if (version != FormatVersion)
        throw std::runtime_error("prepared shard version is unsupported");
    uint32_t flags = readLittleEndian<uint32_t>(in);
    if (flags & ~7U)
        throw std::runtime_error("prepared shard flags are invalid");
    uint32_t sampleCount          = readLittleEndian<uint32_t>(in);
    uint32_t evalTermCount        = readLittleEndian<uint32_t>(in);
    uint32_t policyCandidateCount = readLittleEndian<uint32_t>(in);
    uint32_t policyTargetCount    = readLittleEndian<uint32_t>(in);
    uint64_t shardOrdinal         = readLittleEndian<uint64_t>(in);
    if (bool(flags & 1U) != bool(evalTermCount) || bool(flags & 2U) != bool(policyCandidateCount)
        || bool(flags & 4U) != bool(policyTargetCount))
        throw std::runtime_error("prepared shard objective flags do not match section counts");
    if (shardOrdinal != expectedShardOrdinal)
        throw std::runtime_error("prepared shard ordinal does not match its manifest position");

    std::array<uint8_t, FingerprintBytes> fingerprintBytes;
    for (uint8_t &byte : fingerprintBytes)
        byte = readByte(in);
    if (!expectedFingerprint.empty() && fingerprintBytes != decodeFingerprint(expectedFingerprint))
        throw std::runtime_error("prepared shard fingerprint does not match the cache manifest");

    std::array<SectionMeta, SectionCount> metadata {};
    for (size_t i = 0; i < SectionCount; i++) {
        metadata[i].byteSize = readLittleEndian<uint64_t>(in);
        metadata[i].crc32    = readLittleEndian<uint32_t>(in);
    }
    validateSectionSizes(metadata,
                         sampleCount,
                         evalTermCount,
                         policyCandidateCount,
                         policyTargetCount);

    uint64_t encodedSize     = static_cast<uint64_t>(in.tellg());
    uint64_t allocationBytes = 0;
    for (size_t i = 0; i < SectionCount; i++) {
        const SectionMeta &section = metadata[i];
        if (section.byteSize > std::numeric_limits<uint64_t>::max() - encodedSize
            || section.byteSize > std::numeric_limits<uint64_t>::max() - allocationBytes)
            throw std::runtime_error("prepared shard encoded size overflows");
        encodedSize += section.byteSize;
        allocationBytes += section.byteSize;
    }
    std::error_code fileSizeError;
    uintmax_t       physicalSize = std::filesystem::file_size(path, fileSizeError);
    if (fileSizeError || physicalSize != encodedSize)
        throw std::runtime_error("prepared shard encoded size does not match the file");
    if (allocationBytes > maxAllocationBytes)
        throw std::runtime_error("prepared shard exceeds the configured allocation limit");

    std::vector<uint8_t>          boardSizes(sampleCount);
    std::vector<uint8_t>          results(sampleCount);
    std::vector<int16_t>          staticEvals(sampleCount);
    std::vector<uint16_t>         bestCandidates(sampleCount);
    std::vector<uint32_t>         evalOffsets(size_t(sampleCount) + 1);
    std::vector<TuneCoeff>        evalTerms(evalTermCount);
    std::vector<uint32_t>         policyOffsets(size_t(sampleCount) + 1);
    std::vector<PolicyCandidate>  policyCandidates(policyCandidateCount);
    std::vector<uint32_t>         policyTargetOffsets(size_t(sampleCount) + 1);
    std::vector<PolicyTargetTerm> policyTargets(policyTargetCount);
    bool                          nativeLittleEndian = isLittleEndianHost();

    {
        CheckedSectionReader reader(in, metadata[BoardSizes], verifyChecksum);
        reader.raw(boardSizes.data(), boardSizes.size());
        reader.finish("board size");
    }
    {
        CheckedSectionReader reader(in, metadata[Results], verifyChecksum);
        reader.raw(results.data(), results.size());
        reader.finish("result");
    }
    {
        CheckedSectionReader reader(in, metadata[StaticEvals], verifyChecksum);
        if (nativeLittleEndian)
            reader.raw(staticEvals.data(), staticEvals.size() * sizeof(staticEvals[0]));
        else
            for (int16_t &value : staticEvals)
                value = signed16(reader.littleEndian<uint16_t>());
        reader.finish("static eval");
    }
    {
        CheckedSectionReader reader(in, metadata[BestCandidates], verifyChecksum);
        if (nativeLittleEndian)
            reader.raw(bestCandidates.data(), bestCandidates.size() * sizeof(bestCandidates[0]));
        else
            for (uint16_t &value : bestCandidates)
                value = reader.littleEndian<uint16_t>();
        reader.finish("best candidate");
    }
    {
        CheckedSectionReader reader(in, metadata[EvalOffsets], verifyChecksum);
        if (nativeLittleEndian)
            reader.raw(evalOffsets.data(), evalOffsets.size() * sizeof(evalOffsets[0]));
        else
            for (uint32_t &value : evalOffsets)
                value = reader.littleEndian<uint32_t>();
        reader.finish("value offset");
    }
    {
        CheckedSectionReader reader(in, metadata[EvalTerms], verifyChecksum);
        if (nativeLittleEndian)
            reader.raw(evalTerms.data(), evalTerms.size() * sizeof(evalTerms[0]));
        else
            for (TuneCoeff &term : evalTerms) {
                term.coeff = signed16(reader.littleEndian<uint16_t>());
                term.index = reader.littleEndian<uint16_t>();
            }
        reader.finish("value term");
    }
    {
        CheckedSectionReader reader(in, metadata[PolicyOffsets], verifyChecksum);
        if (nativeLittleEndian)
            reader.raw(policyOffsets.data(), policyOffsets.size() * sizeof(policyOffsets[0]));
        else
            for (uint32_t &value : policyOffsets)
                value = reader.littleEndian<uint32_t>();
        reader.finish("policy offset");
    }
    {
        CheckedSectionReader reader(in, metadata[PolicyCandidates], verifyChecksum);
        if (nativeLittleEndian)
            reader.raw(policyCandidates.data(),
                       policyCandidates.size() * sizeof(policyCandidates[0]));
        else
            for (PolicyCandidate &candidate : policyCandidates)
                for (ParameterId &index : candidate.indices)
                    index = reader.littleEndian<uint16_t>();
        reader.finish("policy candidate");
    }
    {
        CheckedSectionReader reader(in, metadata[PolicyTargetOffsets], verifyChecksum);
        if (nativeLittleEndian)
            reader.raw(policyTargetOffsets.data(),
                       policyTargetOffsets.size() * sizeof(policyTargetOffsets[0]));
        else
            for (uint32_t &value : policyTargetOffsets)
                value = reader.littleEndian<uint32_t>();
        reader.finish("policy target offset");
    }
    {
        CheckedSectionReader reader(in, metadata[PolicyTargets], verifyChecksum);
        if (nativeLittleEndian)
            reader.raw(policyTargets.data(), policyTargets.size() * sizeof(policyTargets[0]));
        else
            for (PolicyTargetTerm &target : policyTargets) {
                target.candidate = reader.littleEndian<uint16_t>();
                target.weight    = reader.littleEndian<uint16_t>();
            }
        reader.finish("policy target");
    }
    if (in.peek() != std::ios::traits_type::eof())
        throw std::runtime_error("prepared shard contains trailing data");

    return PreparedCorpus::fromSections(std::move(boardSizes),
                                        std::move(results),
                                        std::move(staticEvals),
                                        std::move(bestCandidates),
                                        std::move(evalOffsets),
                                        std::move(evalTerms),
                                        std::move(policyOffsets),
                                        std::move(policyCandidates),
                                        std::move(policyTargetOffsets),
                                        std::move(policyTargets));
}

}  // namespace Tuning
