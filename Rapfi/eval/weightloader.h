/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "../core/iohelper.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <istream>
#include <mutex>

namespace Evaluation {

/// Default empty loading args.
struct EmptyLoadArgs
{
    bool operator==(const EmptyLoadArgs &) const { return true; }
};

/// Base class for a weight loader.
/// @tparam WeightType The type of evaluator weight.
template <typename WeightType_, typename LoadArgs_ = EmptyLoadArgs>
struct WeightLoader
{
    typedef WeightType_ WeightType;
    typedef LoadArgs_   LoadArgs;

    /// Load and construct a weight type from the given input stream and load arguments.
    /// @param is Input stream to load weight from.
    /// @param args Extra loading arguments.
    /// @return Weight pointer if load succeeded, otherwise nullptr.
    virtual std::unique_ptr<WeightType> load(std::istream &is, LoadArgs args) = 0;

    /// Whether this weight loader needs a binary stream.
    /// Default behaviour is true. Only used when loading from istream.
    virtual bool needsBinaryStream() const { return true; }
};

/// A weight loader for plain binary Data.
template <typename WeightType_>
struct PlainBinaryWeightLoader : WeightLoader<WeightType_>
{
    using typename WeightLoader<WeightType_>::WeightType;
    using typename WeightLoader<WeightType_>::LoadArgs;

    std::unique_ptr<WeightType> load(std::istream &is, LoadArgs args) override
    {
        auto weight = std::make_unique<WeightType>();
        is.read(reinterpret_cast<char *>(weight.get()), sizeof(WeightType));
        if (is && is.peek() == std::ios::traits_type::eof())
            return std::move(weight);
        else
            return nullptr;
    }
};

/// Standard NNUE weight format header, contains common information about weights.
struct StandardHeader
{
    uint32_t          archHash;
    std::vector<Rule> supportedRules;
    std::vector<int>  supportedBoardSizes;
    std::string       description;
};

/// Weight loader warpper for parsing standard weight format header.
template <typename BaseLoader>
struct StandardHeaderParserWarpper : BaseLoader
{
    using typename BaseLoader::LoadArgs;
    using typename BaseLoader::WeightType;

    template <typename... Args>
    StandardHeaderParserWarpper(Args... args) : BaseLoader(std::forward<Args>(args)...)
    {}

    void setHeaderValidator(std::function<bool(StandardHeader)> validator)
    {
        headerValidator = std::move(validator);
    }

    std::unique_ptr<WeightType> load(std::istream &is, LoadArgs args) override
    {
        struct RawHeaderData
        {
            uint32_t magic;      // 0xacd8cc6a = crc32("gomoku network weight version 1")
            uint32_t arch_hash;  // architecture hash, which is hash of the network architecture
                                 // (network type, num of channels, layers, ...)
            uint32_t rule_mask;  // applicable rule bitmask (1=gomoku, 2=standard, 4=renju)
            uint32_t boardsize_mask;  // applicable board size bitmask (lsb set at index i means
                                      // board size i+1 is usable for this weight)
            uint32_t desc_len;        // length of desc string (=0 for no description)
        } headerData;

        is.read(reinterpret_cast<char *>(&headerData), sizeof(RawHeaderData));
        if (headerData.magic != 0xacd8cc6a)
            return nullptr;

        // Read or skip description text
        if (headerValidator) {
            std::string description;
            description.resize(headerData.desc_len + 1);
            is.read(description.data(), headerData.desc_len);
            if (!headerValidator(StandardHeader {headerData.arch_hash,
                                                 parseRuleMask(headerData.rule_mask),
                                                 parseBoardSizeMask(headerData.boardsize_mask),
                                                 std::move(description)}))
                return nullptr;
        }
        else {
            is.ignore(headerData.desc_len);
        }

        return BaseLoader::load(is, args);
    }

private:
    std::function<bool(StandardHeader)> headerValidator;

    static std::vector<Rule> parseRuleMask(uint32_t ruleMask)
    {
        std::vector<Rule> rules;
        if (ruleMask & 0x1)
            rules.push_back(FREESTYLE);
        if (ruleMask & 0x2)
            rules.push_back(STANDARD);
        if (ruleMask & 0x4)
            rules.push_back(RENJU);
        return rules;
    }

    static std::vector<int> parseBoardSizeMask(uint32_t boardSizeMask)
    {
        std::vector<int> boardSizes;
        for (int i = 0; i < 32; i++) {
            if ((boardSizeMask >> i) & 0x1)
                boardSizes.push_back(i + 1);
        }
        return boardSizes;
    }
};

/// Weight loader warpper for compressed input stream.
template <typename BaseLoader>
struct CompressedWrapper : BaseLoader
{
    using typename BaseLoader::LoadArgs;
    using typename BaseLoader::WeightType;

    template <typename... Args>
    CompressedWrapper(Compressor::Type compressType, Args... args)
        : BaseLoader(std::forward<Args>(args)...)
        , compressType(compressType)
    {}

    void setEntryName(std::string name) { entryName = name; }

    std::unique_ptr<WeightType> load(std::istream &rawInputStream, LoadArgs loadArgs) override
    {
        Compressor    compressor(rawInputStream, compressType);
        std::istream *is = compressor.openInputStream(entryName);
        if (!is)
            return nullptr;
        return BaseLoader::load(*is, loadArgs);
    }

private:
    Compressor::Type compressType;
    std::string      entryName;
};

/// WeightRegistry is the global registry for loaded weights.
/// Usually each evaluator loads weight from file on its own, however in most case all
/// evaluator loads the same weight and it is very memory comsuming to have multiple
/// weight instance in memory. Weight Registry helps to reuse loaded weight when it is
/// applicable, by holding a pool of all loaded weight.
template <typename WeightLoader>
class WeightRegistry
{
public:
    typedef WeightLoader                Loader;
    typedef typename Loader::WeightType WeightType;
    typedef typename Loader::LoadArgs   LoadArgs;

    /// Loads weight from the given file path and load arguments, using the loader.
    /// If the weight already exists in registry, it reuse the loaded weight.
    /// @return Weight pointer, or nullptr if load failed.
    WeightType *
    loadWeightFromFile(Loader &loader, std::filesystem::path filepath, LoadArgs loadArgs = {});

    /// Unloads a loaded weight.
    void unloadWeight(WeightType *weight);

private:
    struct LoadedWeight
    {
        std::unique_ptr<WeightType> weight;
        size_t                      refCount;
        std::filesystem::path       filepath;
        LoadArgs                    loadArgs;
    };

    std::vector<LoadedWeight> pool;
    std::mutex                poolMutex;
};

template <typename WeightLoader>
inline typename WeightRegistry<WeightLoader>::WeightType *
WeightRegistry<WeightLoader>::loadWeightFromFile(
    typename WeightRegistry<WeightLoader>::Loader  &loader,
    std::filesystem::path                           filepath,
    typename WeightRegistry<WeightLoader>::LoadArgs loadArgs)
{
    std::lock_guard<std::mutex> lock(poolMutex);

    // Find weights in loaded weight pool
    for (auto &w : pool) {
        if (w.filepath == filepath && w.loadArgs == loadArgs) {
            w.refCount++;
            return w.weight.get();
        }
    }

    // If not found, load from file
    std::ios_base::openmode mode = std::ios::in;
    if (loader.needsBinaryStream())
        mode = mode | std::ios::binary;
    std::ifstream fileStream(filepath, mode);

    if (!fileStream.is_open())
        return nullptr;

    // Load weight using weight loader
    auto weight = loader.load(fileStream, loadArgs);

    // If load succeeded, add to pool
    if (weight) {
        pool.push_back({std::move(weight), 1, filepath, std::move(loadArgs)});
        return pool.back().weight.get();
    }
    else
        return nullptr;
}

template <typename WeightLoader>
inline void WeightRegistry<WeightLoader>::unloadWeight(
    typename WeightRegistry<WeightLoader>::WeightType *weight)
{
    std::lock_guard<std::mutex> lock(poolMutex);

    for (size_t i = 0; i < pool.size(); i++) {
        if (pool[i].weight.get() == weight) {
            if (--pool[i].refCount == 0)
                pool.erase(pool.begin() + i);
            return;
        }
    }
}

}  // namespace Evaluation
