/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#pragma once

#include "tunecorpus.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace Tuning {

struct PreparedSourcePath
{
    std::string configuredPath;
    std::string canonicalPath;
};

struct PreparedCacheKey
{
    std::string                     fingerprint;
    std::vector<PreparedSourcePath> sourcePaths;
};

/// Ordered, file-backed storage for prepared corpus shards. Each instance
/// publishes into a generation-unique directory and keeps only shard metadata
/// resident between passes.
class FileBackedCorpus
{
public:
    FileBackedCorpus(const std::filesystem::path &root,
                     const char                  *label,
                     size_t                       maxShardBytes,
                     PreparedCacheKey             cacheKey,
                     bool                         rebuild);
    ~FileBackedCorpus();

    void append(PreparedCorpus &&corpus);
    void publish();

    size_t                         size() const { return sampleCount_; }
    bool                           empty() const { return sampleCount_ == 0; }
    size_t                         shardCount() const { return shardCount_; }
    size_t                         diskBytes() const { return diskBytes_; }
    size_t                         maxShardStorageBytes() const { return maxShardStorageBytes_; }
    const std::array<size_t, 256> &boardSampleCounts() const { return boardSampleCounts_; }
    bool                           reused() const { return reused_; }
    const std::string             &cacheStatus() const { return cacheStatus_; }

    PreparedCorpus load(size_t shardIndex) const;

    const std::filesystem::path &directory() const { return directory_; }

private:
    struct ShardInfo
    {
        size_t samples;
        size_t fileBytes;
        size_t storageBytes;
    };

    bool                    tryReuse();
    std::filesystem::path   shardPath(size_t shardIndex) const;
    std::filesystem::path   root_;
    std::string             label_;
    std::filesystem::path   manifestPath_;
    std::filesystem::path   directory_;
    size_t                  maxShardBytes_;
    PreparedCacheKey        cacheKey_;
    std::string             cacheStatus_;
    bool                    reused_               = false;
    bool                    published_            = false;
    size_t                  shardCount_           = 0;
    size_t                  sampleCount_          = 0;
    size_t                  diskBytes_            = 0;
    size_t                  maxShardStorageBytes_ = 0;
    std::array<size_t, 256> boardSampleCounts_    = {};
    std::vector<ShardInfo>  shards_;
    mutable std::vector<uint8_t> validatedShards_;
};

}  // namespace Tuning
