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

#include "../core/platform.h"
#include "../core/pos.h"
#include "../core/types.h"

#include <atomic>
#include <istream>
#include <optional>
#include <ostream>

namespace Search {

/// HashTable class is the shared transposition table implementation
/// with a five-tier bucket system replacement strategies.
class HashTable
{
private:
    static constexpr int CACHE_LINE_SIZE    = 64;
    static constexpr int ENTRIES_PER_BUCKET = 5;

    /// TTEntry struct is a single entry in the transposition table.
    /// To achieve the maximum space efficiency, each TTEntry struct
    /// is compactly stored, using 12 bytes:
    ///     key32        32 bit     (lower 32bit of zobrist key xor data)
    ///     value        16 bit     (value of search)
    ///     eval         16 bit     (value of static evaluation)
    ///     pvNode        1 bit     (is this node pv)
    ///     bound         2 bit     (the bound of search value)
    ///     best         13 bit     (best move only uses the lower 10 bits)
    ///     depth         8 bit     (depth in the search)
    ///     generation    8 bit     (used to find a best replacement entry)
    struct TTEntry
    {
        uint32_t key32;
        union {
            struct
            {
                int16_t  value16;
                int16_t  eval16;
                uint16_t pvBoundBest16;
                uint8_t  depth8;
                uint8_t  generation8;
            };
            uint32_t data[2];
        };

        uint32_t key() const { return key32 ^ data[0] ^ data[1]; }
    };

    /// Bucket Struct contains 3 TTEntry (32 bytes).
    /// Bucket should have a size that can be fitted into one cache line.
    alignas(CACHE_LINE_SIZE) struct Bucket
    {
        TTEntry entry[ENTRIES_PER_BUCKET];
        char    _padding[4];
    } * table;
    size_t  numBuckets;
    uint8_t generation;

    static_assert(sizeof(TTEntry) == 12);
    static_assert(CACHE_LINE_SIZE % sizeof(Bucket) == 0, "Bucket not fitted into cache line");

public:
    HashTable(size_t hashSizeKB);
    ~HashTable();

    /// Get address of the first entry for a hash key.
    TTEntry *firstEntry(HashKey key) const { return table[mulhi64(key, numBuckets)].entry; }
    /// Prefetch the cacheline at the address of a hash key.
    void prefetch(HashKey key) const { ::prefetch(firstEntry(key)); }
    /// Increase the current generation (aging all entries in the table).
    void incGeneration() { generation += 1; }
    /// Resize the tt table to the given hash size in KiB.
    /// If size changed, all hash entries will be cleared after resizing the table.
    /// When memory allocation failed, it will try to find the max available hash
    /// size by reducing cluster count to half recursively.
    void resize(size_t hashSizeKB);
    /// Clear all hash entries. If multi-threading is enabled, clearing will be
    /// performed in parallel with number of threads equals to `Threads.size()`.
    void clear();
    /// Probe the transposition table for a hash key.
    /// @return True if found a matched entry or a not used entry.
    bool probe(HashKey hashKey,
               Value  &ttValue,
               Value  &ttEval,
               bool   &ttIsPv,
               Bound  &ttBound,
               Pos    &ttMove,
               int    &ttDepth,
               int     ply);
    /// Store a new entry in the transposition table.
    void store(HashKey hashKey,
               Value   value,
               Value   eval,
               bool    isPv,
               Bound   bound,
               Pos     move,
               int     depth,
               int     ply);
    /// Dump the transposition table to an ostream.
    void dump(std::ostream &out) const;
    /// Load the transposition table from the stream, previous TT will be
    /// released. Incorrect data stream will cause loading to fail.
    /// @return Whether loading succeeded.
    bool load(std::istream &in);
    /// Estimate the occupation ratio of the tt table during a search.
    /// @return A percentage representing the hash is x permill full.
    int hashUsage() const;
    /// Return the memory usage of the transposition table in KiB.
    size_t hashSizeKB() const { return numBuckets * sizeof(Bucket) / 1024; }
};

extern HashTable TT;

/// PolicyCacheTable class is the shared policy cache table implementation
/// which is used to cache the policy value of good moves in a position.
class PolicyCacheTable
{
public:
    struct PolicyEntry
    {
        static constexpr int MAX_MOVES_PER_ENTRY = 62;

        std::atomic<uint32_t> key32;
        std::atomic_bool      lock;
        uint8_t               numMoves;
        uint8_t               depth;
        uint8_t               generation;

        struct Move
        {
            Pos   pos;
            Score score;
        } moves[MAX_MOVES_PER_ENTRY];

        /// Try lock this policy entry, returns true if lock succeeded.
        bool tryLock()
        {
            return !lock.load(std::memory_order_relaxed)  // prevent excessive coherency traffic
                   && !lock.exchange(true, std::memory_order_acquire);
        }
        /// Unlock this policy entry.
        void unlock() { return lock.store(false, std::memory_order_release); }
        /// Check if the given depth and generation is more valuable than this policy entry.
        bool checkReplaceable(uint8_t depth8, uint8_t currentGeneration) const
        {
            return depth8 > (depth - int(currentGeneration - generation));
        }
    };
    static_assert(sizeof(PolicyEntry) == 256);

    PolicyCacheTable(size_t sizeKB);
    ~PolicyCacheTable();

    /// Resize the tt table to the given hash size in KiB.
    /// If size changed, all hash entries will be cleared after resizing the table.
    /// When memory allocation failed, it will try to find the max available hash
    /// size by reducing cluster count to half recursively.
    void resize(size_t sizeKB);
    /// Clear all hash entries. If multi-threading is enabled, clearing will be
    /// performed in parallel with number of threads equals to `Threads.size()`.
    void clear();
    /// Probe the policy cache table for a hash key.
    /// @return True if found a matched entry or a not used entry.
    bool probe(HashKey key, PolicyEntry *&entry)
    {
        entry = &table[mulhi64(key, numEntries)];
        return entry->key32.load(std::memory_order_relaxed) == uint32_t(key);
    }
    /// Get the current generation.
    uint8_t getGeneration() const { return generation; }
    /// Increase the current generation (aging all entries in the table).
    void incGeneration() { generation += 1; }
    /// Return the memory usage of the policy cache table in KiB.
    size_t hashSizeKB() const { return numEntries * sizeof(PolicyEntry) / 1024; }

private:
    PolicyEntry *table;
    size_t       numEntries;
    uint8_t      generation;
};

extern PolicyCacheTable PCT;

}  // namespace Search
