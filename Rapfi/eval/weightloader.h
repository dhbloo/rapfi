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

#include "../core/compressor.h"
#include "../core/iohelper.h"
#include "../core/platform.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <istream>
#include <mutex>
#include <type_traits>

namespace Evaluation {

/// Linear scan: does `c` contain a value equal to `elem`? Used by the weight loaders below to
/// validate header fields (supported rules / board sizes) and model output names.
template <typename Container, typename ElemType>
bool contains(const Container &c, ElemType elem)
{
    for (auto it = c.cbegin(); it != c.end(); it++) {
        if (*it == elem)
            return true;
    }
    return false;
}

/// Default empty loading args.
struct EmptyLoadArgs
{
    bool operator==(const EmptyLoadArgs &) const { return true; }
};

/// Base class for a weight loader.
/// Loaders compose statically by wrapping (e.g. CompressedWrapper<StandardHeaderLoader<X>>),
/// so this base is used as a concept that pins WeightType/LoadArgs; concrete loaders shadow
/// load() rather than override it, and no loader is ever called through this base.
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
    virtual LargePagePtr<WeightType> load(std::istream &is, LoadArgs args) = 0;

    /// Whether this weight loader needs a binary stream.
    /// Default behaviour is true. Only used when loading from istream.
    virtual bool needsBinaryStream() const { return true; }
};

/// Standard NNUE weight format header, contains common information about weights.
struct StandardHeader
{
    uint32_t          archHash;
    std::vector<Rule> supportedRules;
    std::vector<int>  supportedBoardSizes;
    std::string       description;
};

/// Load arguments for standard header weight loader.
template <typename BaseLoader>
struct StandardHeaderParseLoadArgs : public BaseLoader::LoadArgs
{
    int                   boardSize;
    Rule                  rule;
    std::filesystem::path weightPath;

    bool operator==(const StandardHeaderParseLoadArgs &other) const
    {
        return rule == other.rule && boardSize == other.boardSize && weightPath == other.weightPath
               && BaseLoader::LoadArgs::operator==(other);
    }
};

/// Weight loader for parsing standard weight format header.
/// This loader wraps another weight loader and adds support for parsing the standard header.
template <typename BaseLoader>
struct StandardHeaderLoader
    : public WeightLoader<typename BaseLoader::WeightType, StandardHeaderParseLoadArgs<BaseLoader>>
{
    using WeightType = typename BaseLoader::WeightType;
    using LoadArgs   = StandardHeaderParseLoadArgs<BaseLoader>;

    StandardHeaderLoader() = default;

    void setHeaderValidator(std::function<bool(StandardHeader, LoadArgs &)> validator)
    {
        headerValidator = std::move(validator);
    }

    LargePagePtr<WeightType> load(std::istream &is, LoadArgs args) override
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
        if (!is || headerData.magic != 0xacd8cc6a)
            return nullptr;

        // Bound allocations from a corrupt header before reading its description.
        constexpr uint32_t MaxDescLen = 1048576;
        if (headerData.desc_len > MaxDescLen)
            return nullptr;

        // Read or skip description text
        if (headerValidator) {
            std::string description;
            description.resize(static_cast<size_t>(headerData.desc_len) + 1);
            is.read(description.data(), headerData.desc_len);
            if (!is)
                return nullptr;
            auto header = StandardHeader {headerData.arch_hash,
                                          parseRuleMask(headerData.rule_mask),
                                          parseBoardSizeMask(headerData.boardsize_mask),
                                          std::move(description)};
            if (!headerValidator(header, args))
                return nullptr;
        }
        else {
            is.ignore(headerData.desc_len);
            if (!is || is.gcount() != headerData.desc_len)
                return nullptr;
        }

        return baseLoader.load(is, args);
    }

private:
    BaseLoader                                      baseLoader;
    std::function<bool(StandardHeader, LoadArgs &)> headerValidator;

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

/// Weight loader wrapper for compressed input stream.
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

    LargePagePtr<WeightType> load(std::istream &rawInputStream, LoadArgs loadArgs) override
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
/// applicable, by holding a weightPool of all loaded weight.
template <typename WeightLoader>
class WeightRegistry
{
public:
    typedef WeightLoader                Loader;
    typedef typename Loader::WeightType WeightType;
    typedef typename Loader::LoadArgs   LoadArgs;

    /// Loads weight from the given file path and load arguments, using the loader.
    /// If the weight already exists in registry, it reuse the loaded weight.
    /// @param loader Loader to use for loading the weight.
    /// @param filepath Path to the weight file.
    /// @param numaNodeId Numa node ID of the current thread. If the weight has not been loaded
    ///      on this NUMA node, it will be copyed to the current NUMA node using first-touch policy.
    /// @param loadArgs Extra loading arguments for the weight loader.
    /// @return Weight pointer, or nullptr if load failed.
    WeightType *loadWeightFromFile(Loader               &loader,
                                   std::filesystem::path filepath,
                                   Numa::NumaNodeId      numaNodeId,
                                   LoadArgs              loadArgs = {});

    /// Unloads a loaded weight.
    void unloadWeight(WeightType *weight);

private:
    struct LoadedWeight
    {
        LargePagePtr<WeightType> weight;      // Pointer to the loaded weight
        size_t                   refCount;    // Reference count of the loaded weight
        std::filesystem::path    filepath;    // File path from which the weight was loaded
        Numa::NumaNodeId         numaNodeId;  // Numa node ID where the weight was loaded
        LoadArgs                 loadArgs;    // Extra loading arguments used for loading the weight
    };

    /// Pool of loaded weights.
    /// Each weight is stored with its reference count, file path, NUMA node ID and load arguments.
    std::vector<LoadedWeight> weightPool;

    /// Mutex to protect concurrent access to the weight pool.
    std::mutex poolMutex;
};

template <typename WeightLoader>
inline typename WeightRegistry<WeightLoader>::WeightType *
WeightRegistry<WeightLoader>::loadWeightFromFile(
    typename WeightRegistry<WeightLoader>::Loader  &loader,
    std::filesystem::path                           filepath,
    Numa::NumaNodeId                                numaNodeId,
    typename WeightRegistry<WeightLoader>::LoadArgs loadArgs)
{
    std::lock_guard<std::mutex> lock(poolMutex);

    // Reuse a weight already loaded on the current NUMA node before making another copy.
    for (auto &w : weightPool) {
        if (w.filepath == filepath && w.loadArgs == loadArgs && w.numaNodeId == numaNodeId) {
            w.refCount++;
            return w.weight.get();
        }
    }

    // Copy a remote weight using first-touch allocation on the current NUMA node.
    for (auto &w : weightPool) {
        if (w.filepath == filepath && w.loadArgs == loadArgs) {
            LargePagePtr<WeightType> copiedWeight {nullptr};
            if constexpr (std::is_copy_constructible_v<WeightType>)
                copiedWeight = make_unique_large_page<WeightType>(*w.weight);

            // Non-copyable weights are shared across NUMA nodes.
            if (copiedWeight) {
                auto copiedWeightPtr = copiedWeight.get();
                weightPool.push_back(
                    {std::move(copiedWeight), 1, filepath, numaNodeId, std::move(loadArgs)});
                return copiedWeightPtr;
            }

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
    if (!weight)
        return nullptr;

    // If load succeeded, add to weightPool
    auto weightPtr = weight.get();
    weightPool.push_back({std::move(weight), 1, filepath, numaNodeId, std::move(loadArgs)});
    return weightPtr;
}

template <typename WeightLoader>
inline void WeightRegistry<WeightLoader>::unloadWeight(
    typename WeightRegistry<WeightLoader>::WeightType *weight)
{
    std::lock_guard<std::mutex> lock(poolMutex);

    for (size_t i = 0; i < weightPool.size(); i++) {
        if (weightPool[i].weight.get() == weight) {
            if (--weightPool[i].refCount == 0)
                weightPool.erase(weightPool.begin() + i);
            return;
        }
    }
}

}  // namespace Evaluation
