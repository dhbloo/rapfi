/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "tunestore.h"

#include "tuneshard.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <system_error>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

namespace Tuning {
namespace {

    constexpr uint32_t    ManifestVersion         = 6;
    constexpr uint32_t    GenerationMarkerVersion = 1;
    constexpr const char *GenerationMarkerName    = ".rftune-generation";
    constexpr size_t      MaxManifestBytes        = 8 * 1024 * 1024;

    class ReadOnlyMemoryBuffer : public std::streambuf
    {
    public:
        explicit ReadOnlyMemoryBuffer(std::string &contents)
        {
            if (contents.empty())
                setg(nullptr, nullptr, nullptr);
            else {
                char *begin = &contents[0];
                setg(begin, begin, begin + contents.size());
            }
        }
    };

    std::filesystem::path makeGenerationDirectory(const std::filesystem::path &root,
                                                  const std::string           &label)
    {
        static std::atomic<uint64_t> sequence {0};
        uint64_t                     timestamp = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        for (uint64_t attempt = 0; attempt < 1024; attempt++) {
            std::ostringstream name;
            name << label << '-' << std::hex << timestamp << '-'
                 << (sequence.fetch_add(1, std::memory_order_relaxed) + attempt);
            std::filesystem::path candidate = root / name.str();
            std::error_code       error;
            if (std::filesystem::create_directories(candidate, error))
                return candidate;
            if (error && error != std::errc::file_exists)
                throw std::runtime_error("unable to create prepared corpus directory: "
                                         + error.message());
        }
        throw std::runtime_error("unable to allocate a unique prepared corpus generation");
    }

    std::string shardFilename(size_t shardIndex)
    {
        std::ostringstream filename;
        filename << "shard-" << std::setfill('0') << std::setw(8) << shardIndex << ".rftune";
        return filename.str();
    }

    bool isSafeGenerationName(const std::string &name, const std::string &label)
    {
        std::filesystem::path path(name);
        std::string           prefix = label + '-';
        if (name.empty() || path != path.filename() || name.compare(0, prefix.size(), prefix) != 0)
            return false;
        size_t separator = name.find('-', prefix.size());
        if (separator == std::string::npos || separator == prefix.size()
            || separator + 1 == name.size() || name.find('-', separator + 1) != std::string::npos)
            return false;
        auto isHexRange = [&](size_t begin, size_t end) {
            return std::all_of(name.begin() + begin, name.begin() + end, [](char digit) {
                return (digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f');
            });
        };
        return isHexRange(prefix.size(), separator) && isHexRange(separator + 1, name.size());
    }

    bool isDirectOwnedDirectory(const std::filesystem::path &root,
                                const std::filesystem::path &directory)
    {
        std::error_code error;
        auto            status = std::filesystem::symlink_status(directory, error);
        if (error || !std::filesystem::is_directory(status) || std::filesystem::is_symlink(status))
            return false;
        std::filesystem::path canonicalRoot = std::filesystem::weakly_canonical(root, error);
        if (error)
            return false;
        std::filesystem::path canonicalDirectory =
            std::filesystem::weakly_canonical(directory, error);
        return !error && canonicalDirectory.parent_path() == canonicalRoot;
    }

    size_t checkedSize(uint64_t value, const char *field)
    {
        if (value > std::numeric_limits<size_t>::max())
            throw std::runtime_error(std::string("prepared manifest ") + field + " exceeds size_t");
        return static_cast<size_t>(value);
    }

    void expectToken(std::istream &input, const char *expected)
    {
        std::string token;
        if (!(input >> token) || token != expected)
            throw std::runtime_error(std::string("prepared manifest expected '") + expected + "'");
    }

    void writeGenerationMarker(const std::filesystem::path &directory,
                               const std::string           &label,
                               const std::string           &fingerprint)
    {
        std::ofstream output(directory / GenerationMarkerName, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("unable to create prepared generation marker");
        output << "RFTUNE_GENERATION " << GenerationMarkerVersion << '\n';
        output << "label " << std::quoted(label) << '\n';
        output << "generation " << std::quoted(directory.filename().string()) << '\n';
        output << "fingerprint " << fingerprint << '\n';
        output << "end\n";
        output.close();
        if (!output)
            throw std::runtime_error("failed to publish prepared generation marker");
    }

    bool hasMatchingGenerationMarker(const std::filesystem::path &root,
                                     const std::filesystem::path &directory,
                                     const std::string           &label,
                                     const std::string           &fingerprint)
    {
        if (!isSafeGenerationName(directory.filename().string(), label)
            || !isDirectOwnedDirectory(root, directory))
            return false;
        std::ifstream input(directory / GenerationMarkerName, std::ios::binary);
        if (!input)
            return false;
        try {
            expectToken(input, "RFTUNE_GENERATION");
            uint32_t version;
            if (!(input >> version) || version != GenerationMarkerVersion)
                return false;
            expectToken(input, "label");
            std::string actualLabel;
            if (!(input >> std::quoted(actualLabel)) || actualLabel != label)
                return false;
            expectToken(input, "generation");
            std::string generation;
            if (!(input >> std::quoted(generation)) || generation != directory.filename().string())
                return false;
            expectToken(input, "fingerprint");
            std::string actualFingerprint;
            if (!(input >> actualFingerprint) || actualFingerprint != fingerprint)
                return false;
            expectToken(input, "end");
            std::string trailing;
            return !(input >> trailing);
        }
        catch (...) {
            return false;
        }
    }

    std::string readManifest(const std::filesystem::path &path)
    {
        std::error_code error;
        uintmax_t       fileSize = std::filesystem::file_size(path, error);
        if (error)
            throw std::runtime_error("unable to inspect prepared manifest: " + error.message());
        if (fileSize > MaxManifestBytes)
            throw std::runtime_error("prepared manifest exceeds its memory allowance");

        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("unable to open prepared manifest");
        std::string contents(static_cast<size_t>(fileSize), '\0');
        if (!contents.empty())
            input.read(&contents[0], static_cast<std::streamsize>(contents.size()));
        if (input.gcount() != static_cast<std::streamsize>(contents.size())
            || input.peek() != std::ios::traits_type::eof())
            throw std::runtime_error("prepared manifest changed while being read");
        return contents;
    }

    void atomicReplace(const std::filesystem::path &source, const std::filesystem::path &target)
    {
#ifdef _WIN32
        DWORD lastError = ERROR_SUCCESS;
        for (int attempt = 0; attempt < 1000; attempt++) {
            if (MoveFileExW(source.c_str(),
                            target.c_str(),
                            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                return;
            lastError = GetLastError();
            if (lastError != ERROR_ACCESS_DENIED && lastError != ERROR_SHARING_VIOLATION)
                break;
            Sleep(5);
        }
        std::error_code error(static_cast<int>(lastError), std::system_category());
        throw std::runtime_error("failed to publish prepared manifest: " + error.message());
#else
        std::error_code error;
        std::filesystem::rename(source, target, error);
        if (error)
            throw std::runtime_error("failed to publish prepared manifest: " + error.message());
#endif
    }

}  // namespace

FileBackedCorpus::FileBackedCorpus(const std::filesystem::path &root,
                                   const char                  *label,
                                   size_t                       maxShardBytes,
                                   PreparedCacheKey             cacheKey,
                                   bool                         rebuild)
    : root_(root)
    , label_(label)
    , manifestPath_(root_ / (label_ + ".manifest"))
    , maxShardBytes_(maxShardBytes)
    , cacheKey_(std::move(cacheKey))
{
    if (cacheKey_.fingerprint.size() != 64)
        throw std::invalid_argument("prepared cache fingerprint must contain 64 hex digits");
    std::filesystem::create_directories(root_);

    if (!rebuild && tryReuse())
        return;
    if (rebuild)
        cacheStatus_ = "forced prepared-cache rebuild";
    directory_ = makeGenerationDirectory(root_, label_);
    try {
        writeGenerationMarker(directory_, label_, cacheKey_.fingerprint);
    }
    catch (...) {
        std::error_code ignored;
        std::filesystem::remove_all(directory_, ignored);
        directory_.clear();
        throw;
    }
}

FileBackedCorpus::~FileBackedCorpus()
{
    if (!reused_ && !published_ && !directory_.empty()
        && hasMatchingGenerationMarker(root_, directory_, label_, cacheKey_.fingerprint)) {
        std::error_code ignored;
        std::filesystem::remove_all(directory_, ignored);
    }
}

bool FileBackedCorpus::tryReuse()
{
    std::ifstream manifestProbe(manifestPath_, std::ios::binary);
    if (!manifestProbe) {
        cacheStatus_ = "prepared-cache miss: manifest not found";
        return false;
    }
    manifestProbe.close();

    try {
        std::string          manifestContents = readManifest(manifestPath_);
        ReadOnlyMemoryBuffer manifestBuffer(manifestContents);
        std::istream         input(&manifestBuffer);
        expectToken(input, "RFTUNE_MANIFEST");
        uint32_t version;
        if (!(input >> version) || version != ManifestVersion)
            throw std::runtime_error("prepared manifest version is unsupported");
        expectToken(input, "fingerprint");
        std::string fingerprint;
        if (!(input >> fingerprint))
            throw std::runtime_error("prepared manifest fingerprint is missing");
        expectToken(input, "generation");
        std::string generation;
        if (!(input >> std::quoted(generation)) || !isSafeGenerationName(generation, label_))
            throw std::runtime_error("prepared manifest generation is invalid");
        if (fingerprint != cacheKey_.fingerprint) {
            cacheStatus_ = "prepared-cache miss: prepared inputs changed";
            return false;
        }
        std::filesystem::path generationDirectory = root_ / generation;
        if (!hasMatchingGenerationMarker(root_, generationDirectory, label_, cacheKey_.fingerprint))
            throw std::runtime_error("prepared generation ownership marker is invalid");

        expectToken(input, "shards");
        uint64_t declaredShardCount;
        if (!(input >> declaredShardCount))
            throw std::runtime_error("prepared manifest shard count is missing");
        expectToken(input, "samples");
        uint64_t declaredSamples;
        if (!(input >> declaredSamples))
            throw std::runtime_error("prepared manifest sample count is missing");
        expectToken(input, "disk_bytes");
        uint64_t declaredDiskBytes;
        if (!(input >> declaredDiskBytes))
            throw std::runtime_error("prepared manifest disk size is missing");
        expectToken(input, "max_shard_storage");
        uint64_t declaredMaxStorage;
        if (!(input >> declaredMaxStorage))
            throw std::runtime_error("prepared manifest maximum shard size is missing");
        expectToken(input, "boards");
        uint64_t declaredBoardCount;
        if (!(input >> declaredBoardCount) || declaredBoardCount > boardSampleCounts_.size())
            throw std::runtime_error("prepared manifest board count is invalid");
        std::array<size_t, 256> validatedBoardCounts {};
        size_t                  boardSamples = 0;
        for (uint64_t i = 0; i < declaredBoardCount; i++) {
            expectToken(input, "board");
            uint64_t boardSize, samples;
            if (!(input >> boardSize >> samples) || boardSize >= validatedBoardCounts.size()
                || validatedBoardCounts[boardSize] != 0)
                throw std::runtime_error("prepared manifest board record is invalid");
            size_t sampleCount = checkedSize(samples, "board sample count");
            if (sampleCount == 0 || sampleCount > std::numeric_limits<size_t>::max() - boardSamples)
                throw std::runtime_error("prepared manifest board sample count is invalid");
            validatedBoardCounts[boardSize] = sampleCount;
            boardSamples += sampleCount;
        }

        size_t validatedShards     = checkedSize(declaredShardCount, "shard count");
        if (validatedShards > manifestContents.size())
            throw std::runtime_error("prepared manifest shard count is invalid");
        size_t validatedSamples    = 0;
        size_t validatedDiskBytes  = 0;
        size_t validatedMaxStorage = 0;
        std::vector<ShardInfo> validatedShardInfo;
        validatedShardInfo.reserve(validatedShards);
        for (size_t shard = 0; shard < validatedShards; shard++) {
            expectToken(input, "shard");
            std::string filename;
            uint64_t    ordinal, declaredShardSamples, declaredFileBytes, declaredStorageBytes;
            if (!(input >> std::quoted(filename) >> ordinal >> declaredShardSamples
                  >> declaredFileBytes >> declaredStorageBytes)
                || filename != shardFilename(shard) || ordinal != shard)
                throw std::runtime_error("prepared manifest shard order is invalid");
            std::filesystem::path path = generationDirectory / filename;
            std::error_code       fileError;
            uintmax_t             fileSize = std::filesystem::file_size(path, fileError);
            if (fileError || fileSize != declaredFileBytes)
                throw std::runtime_error("prepared manifest shard file size does not match");
            ShardInfo info {checkedSize(declaredShardSamples, "shard sample count"),
                            checkedSize(declaredFileBytes, "shard file size"),
                            checkedSize(declaredStorageBytes, "shard storage size")};
            if (info.storageBytes > maxShardBytes_
                || info.samples > std::numeric_limits<size_t>::max() - validatedSamples
                || info.fileBytes > std::numeric_limits<size_t>::max() - validatedDiskBytes)
                throw std::runtime_error("prepared cache aggregate size overflows");
            validatedSamples += info.samples;
            validatedDiskBytes += info.fileBytes;
            validatedMaxStorage = std::max(validatedMaxStorage, info.storageBytes);
            validatedShardInfo.push_back(info);
        }
        expectToken(input, "end");
        std::string trailing;
        if (input >> trailing)
            throw std::runtime_error("prepared manifest contains trailing fields");

        if (validatedSamples != checkedSize(declaredSamples, "sample count")
            || validatedDiskBytes != checkedSize(declaredDiskBytes, "disk size")
            || validatedMaxStorage != checkedSize(declaredMaxStorage, "maximum shard size")
            || boardSamples != validatedSamples)
            throw std::runtime_error("prepared manifest aggregate metadata does not match shards");

        directory_             = std::move(generationDirectory);
        shardCount_           = validatedShards;
        sampleCount_          = validatedSamples;
        diskBytes_            = validatedDiskBytes;
        maxShardStorageBytes_ = validatedMaxStorage;
        boardSampleCounts_    = validatedBoardCounts;
        shards_               = std::move(validatedShardInfo);
        validatedShards_.assign(shardCount_, uint8_t(0));
        reused_               = true;
        published_            = true;
        cacheStatus_          = "reused prepared cache";
        return true;
    }
    catch (const std::exception &error) {
        directory_.clear();
        boardSampleCounts_.fill(0);
        shards_.clear();
        validatedShards_.clear();
        cacheStatus_ = std::string("prepared-cache rebuild: ") + error.what();
        return false;
    }
}

std::filesystem::path FileBackedCorpus::shardPath(size_t shardIndex) const
{
    return directory_ / shardFilename(shardIndex);
}

void FileBackedCorpus::append(PreparedCorpus &&corpus)
{
    if (reused_ || published_)
        throw std::logic_error("cannot append to a published prepared corpus");
    if (corpus.empty())
        return;
    if (sampleCount_ > std::numeric_limits<size_t>::max() - corpus.size())
        throw std::length_error("prepared corpus sample count overflows size_t");

    size_t samples      = corpus.size();
    size_t storageBytes = corpus.storageBytes();
    if (storageBytes > maxShardBytes_)
        throw std::runtime_error("prepared shard exceeds its allocation credit");
    for (uint8_t boardSize : corpus.boardSizes())
        boardSampleCounts_[boardSize]++;
    std::filesystem::path path = shardPath(shardCount_);
    writePreparedShard(path, corpus, cacheKey_.fingerprint, shardCount_);

    uintmax_t fileSize = std::filesystem::file_size(path);
    if (fileSize > std::numeric_limits<size_t>::max())
        throw std::length_error("prepared shard file size exceeds size_t");
    size_t fileBytes = static_cast<size_t>(fileSize);
    if (diskBytes_ > std::numeric_limits<size_t>::max() - fileBytes)
        throw std::length_error("prepared corpus disk byte count overflows size_t");

    shards_.push_back({samples, fileBytes, storageBytes});
    validatedShards_.push_back(uint8_t(0));
    corpus = PreparedCorpus {};
    shardCount_++;
    sampleCount_ += samples;
    diskBytes_ += fileBytes;
    maxShardStorageBytes_ = std::max(maxShardStorageBytes_, storageBytes);
}

void FileBackedCorpus::publish()
{
    if (reused_ || published_)
        return;
    if (shards_.size() != shardCount_ || validatedShards_.size() != shardCount_)
        throw std::logic_error("prepared corpus shard metadata is incomplete");

    std::filesystem::path tempPath = directory_ / "manifest.tmp";
    try {
        if (std::filesystem::exists(tempPath))
            throw std::runtime_error("prepared manifest temporary target already exists");
        std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("unable to create prepared manifest: " + tempPath.string());
        output << "RFTUNE_MANIFEST " << ManifestVersion << '\n';
        output << "fingerprint " << cacheKey_.fingerprint << '\n';
        output << "generation " << std::quoted(directory_.filename().string()) << '\n';
        output << "shards " << shardCount_ << '\n';
        output << "samples " << sampleCount_ << '\n';
        output << "disk_bytes " << diskBytes_ << '\n';
        output << "max_shard_storage " << maxShardStorageBytes_ << '\n';
        size_t activeBoards = std::count_if(boardSampleCounts_.begin(),
                                            boardSampleCounts_.end(),
                                            [](size_t count) { return count != 0; });
        output << "boards " << activeBoards << '\n';
        for (size_t boardSize = 0; boardSize < boardSampleCounts_.size(); boardSize++)
            if (boardSampleCounts_[boardSize] != 0)
                output << "board " << boardSize << ' ' << boardSampleCounts_[boardSize] << '\n';
        for (size_t shard = 0; shard < shardCount_; shard++) {
            const ShardInfo &info = shards_[shard];
            output << "shard " << std::quoted(shardFilename(shard)) << ' ' << shard << ' '
                   << info.samples << ' ' << info.fileBytes << ' ' << info.storageBytes << '\n';
        }

        output << "end\n";
        output.flush();
        if (!output)
            throw std::runtime_error("failed to write prepared manifest: " + tempPath.string());
        output.close();
        if (!output)
            throw std::runtime_error("failed to close prepared manifest: " + tempPath.string());
        std::error_code manifestSizeError;
        uintmax_t       manifestSize = std::filesystem::file_size(tempPath, manifestSizeError);
        if (manifestSizeError)
            throw std::runtime_error("unable to inspect completed prepared manifest: "
                                     + manifestSizeError.message());
        if (manifestSize > MaxManifestBytes)
            throw std::runtime_error("prepared manifest exceeds its reusable memory allowance; "
                                     "increase --shard-size-mb or reduce the source-file count");

        atomicReplace(tempPath, manifestPath_);
        published_ = true;
    }
    catch (...) {
        std::error_code ignored;
        std::filesystem::remove(tempPath, ignored);
        throw;
    }
}

PreparedCorpus FileBackedCorpus::load(size_t shardIndex) const
{
    if (shardIndex >= shardCount_)
        throw std::out_of_range("prepared corpus shard index is out of range");
    bool verifyChecksum = validatedShards_[shardIndex] == 0;
    PreparedCorpus corpus = readPreparedShard(shardPath(shardIndex),
                                               verifyChecksum,
                                               maxShardBytes_,
                                               cacheKey_.fingerprint,
                                               shardIndex);
    const ShardInfo &expected = shards_[shardIndex];
    if (corpus.size() != expected.samples || corpus.storageBytes() != expected.storageBytes)
        throw std::runtime_error("prepared corpus shard metadata does not match its manifest");
    validatedShards_[shardIndex] = uint8_t(1);
    return corpus;
}

}  // namespace Tuning
