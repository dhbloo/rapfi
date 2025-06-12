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

#include "hashtable.h"

#include "../core/iohelper.h"
#include "../core/platform.h"
#include "../core/utils.h"
#include "searchthread.h"

#include <cassert>
#include <cstring>  // For std::memset
#include <vector>
#ifdef MULTI_THREADING
    #include <thread>
#endif

static const char HashDumpMagicString[32] = "RAPFI HASH DUMP VER 001";

namespace Search {

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

// Sanity check on TTEntry size
static_assert(sizeof(TTEntry) == 12);

/// TTBucket Struct contains 5 TTEntry (64 bytes), which should be fitted into one cache line.
struct TTBucket
{
    TTEntry entry[ENTRIES_PER_BUCKET];
    char    _padding[4];
};

// Make sure the size of TTBucket can be fitted into one cache line
static_assert(CACHE_LINE_SIZE % sizeof(TTBucket) == 0, "TTBucket not fitted into cache line");

/// Global shared transposition table
HashTable TT {16 * 1024};  // default size is 16 MB

HashTable::HashTable(size_t hashSizeKB) : table(nullptr), numBuckets(0)
{
    resize(hashSizeKB);
}

HashTable::~HashTable()
{
    MemAlloc::alignedLargePageFree(table);
}

void HashTable::resize(size_t hashSizeKB)
{
    size_t newNumBuckets = hashSizeKB * (1024 / sizeof(TTBucket));
    newNumBuckets        = std::max<size_t>(newNumBuckets, 1);

    if (newNumBuckets == numBuckets)
        return;

    numBuckets = newNumBuckets;

    if (table) {
        Threads.waitForIdle();
        MemAlloc::alignedLargePageFree(table);
        table = nullptr;
    }

    size_t tryNumBuckets = numBuckets;
    while (tryNumBuckets) {
        size_t allocSize = sizeof(TTBucket) * tryNumBuckets;
        table            = static_cast<TTBucket *>(MemAlloc::alignedLargePageAlloc(allocSize));

        if (!table)
            tryNumBuckets /= 2;
        else
            break;
    }

    if (tryNumBuckets != numBuckets) {
        numBuckets = tryNumBuckets;
        ERRORL("Failed to allocate " << hashSizeKB << " KB for transposition table.");

        // Exit program if failed to allocate 1 cluster
        if (!numBuckets)
            std::exit(EXIT_FAILURE);

        MESSAGEL("Allocated " << (numBuckets * sizeof(TTBucket) >> 10)
                              << " KB for transposition table.");
    }

    clear();
}

void HashTable::clear()
{
#if defined(MULTI_THREADING) && !defined(__EMSCRIPTEN__)
    // Clear hash table in a multi-threaded way
    std::vector<std::thread> threads;
    size_t                   numThreads = std::max<size_t>(Threads.size(), 1);
    size_t                   stride     = numBuckets / numThreads;

    for (size_t idx = 0; idx < numThreads; idx++) {
        threads.emplace_back([=]() {
            // Thread binding gives faster search on systems with a first-touch policy
            if (Threads.size() > 8)
                Numa::bindThisThread(idx);

            // Each thread will zero its part of the hash table
            size_t start = stride * idx;
            size_t len   = idx != numThreads - 1 ? stride : numBuckets - start;

            std::memset(&table[start], 0, len * sizeof(TTBucket));
        });
    }

    for (std::thread &th : threads)
        th.join();
#else
    std::memset(table, 0, numBuckets * sizeof(TTBucket));
#endif

    generation = 0;
}

TTEntry *HashTable::firstEntry(HashKey key) const
{
    return table[mulhi64(key, numBuckets)].entry;
}

void HashTable::prefetch(HashKey key) const
{
    ::prefetch(firstEntry(key));
}

bool HashTable::probe(HashKey hashKey,
                      Value  &ttValue,
                      Value  &ttEval,
                      bool   &ttIsPv,
                      Bound  &ttBound,
                      Pos    &ttMove,
                      int    &ttDepth,
                      int     ply)
{
    TTEntry *entry = firstEntry(hashKey);
    uint32_t key32 = uint32_t(hashKey);

    // Iterate the bucket to find a matched entry
    for (int i = 0; i < ENTRIES_PER_BUCKET; i++) {
        TTEntry tte = entry[i];  // Copy tte from shared memory to stack

        if (tte.key() == key32) {
            // Update current entry's generation, as well as key32
            entry[i].generation8 = generation;
            entry[i].key32 ^= tte.data[1] ^ entry[i].data[1];

            ttValue = storedValueToSearchValue(tte.value16, ply);
            ttEval  = Value(tte.eval16);
            ttIsPv  = bool(tte.pvBoundBest16 >> 15);
            ttBound = Bound((tte.pvBoundBest16 >> 13) & 0x3);
            ttMove  = Pos((tte.pvBoundBest16 & 0x3ff) - 1);
            ttDepth = int(tte.depth8) + (int)DEPTH_LOWER_BOUND;

            return true;
        }
    }

    return false;
}

void HashTable::store(HashKey hashKey,
                      Value   value,
                      Value   eval,
                      bool    isPv,
                      Bound   bound,
                      Pos     move,
                      int     depth,
                      int     ply)
{
    TTEntry *entry        = firstEntry(hashKey);
    uint32_t newKey32     = uint32_t(hashKey);
    TTEntry *replace      = &entry[0];
    auto     replaceValue = [=](const TTEntry &e) {
        uint8_t relativeAge = generation - e.generation8;
        return int(e.depth8) - int(relativeAge);
    };

    // Iterate the bucket to find a matched entry or a least valuable entry for replacement
    for (int i = 0; i < ENTRIES_PER_BUCKET; i++) {
        if (entry[i].key() == newKey32) {
            replace = &entry[i];
            break;
        }
        if (replaceValue(entry[i]) < replaceValue(*replace))
            replace = &entry[i];
    }

    uint32_t oldKey32 = replace->key();

    // We only overwrite the same position if the new record has an exact bound
    // or a depth that is nearly as deep as the last one
    if (bound != BOUND_EXACT && newKey32 == oldKey32
        && depth + 2 < replace->depth8 + (int)DEPTH_LOWER_BOUND)
        return;

    assert(value >= VALUE_NONE && value <= VALUE_INFINITE);
    assert(depth > (int)DEPTH_LOWER_BOUND && depth < (int)DEPTH_LOWER_BOUND + 256);

    // Use previous stored best move if we do not have a best move this time
    if (move == Pos::NONE && newKey32 == oldKey32)
        move = Pos((replace->pvBoundBest16 & 0x3ff) - 1);

    // Construct the new tt entry on stack first, then copy it to shared memory
    // Note that best move is stored by adding 1 to offset the Pos::PASS.
    TTEntry newEntry;
    newEntry.value16       = int16_t(searchValueToStoredValue(value, ply));
    newEntry.eval16        = int16_t(eval);
    newEntry.pvBoundBest16 = uint16_t(isPv) << 15 | uint16_t(bound) << 13 | uint16_t((int)move + 1);
    newEntry.depth8        = uint8_t(depth - (int)DEPTH_LOWER_BOUND);
    newEntry.generation8   = uint8_t(generation);
    newEntry.key32         = newKey32 ^ newEntry.data[0] ^ newEntry.data[1];

    *replace = newEntry;  // Copy to shared memory
}

void HashTable::dump(std::ostream &outStream) const
{
    Compressor    compressor(outStream, Compressor::Type::LZ4_DEFAULT);
    std::ostream *out = compressor.openOutputStream();
    assert(out);

    // Write hash dump magic
    out->write(HashDumpMagicString, sizeof(HashDumpMagicString));
    out->write(reinterpret_cast<const char *>(&numBuckets), sizeof(numBuckets));
    out->write(reinterpret_cast<const char *>(&generation), sizeof(generation));

    for (size_t i = 0; i < numBuckets; i++) {
        TTBucket &cluster = table[i];
        out->write(reinterpret_cast<const char *>(&cluster), sizeof(TTBucket));
    }
}

bool HashTable::load(std::istream &inStream)
{
    Compressor    compressor(inStream, Compressor::Type::LZ4_DEFAULT);
    std::istream *in = compressor.openInputStream();
    if (!in)
        return false;

    // Validate hash dump magic
    char magic[sizeof(HashDumpMagicString)];
    in->read(magic, sizeof(HashDumpMagicString));
    if (std::memcmp(magic, HashDumpMagicString, sizeof(HashDumpMagicString)) != 0)
        return false;

    in->read(reinterpret_cast<char *>(&numBuckets), sizeof(numBuckets));
    in->read(reinterpret_cast<char *>(&generation), sizeof(generation));
    if (numBuckets == 0)
        return false;

    if (table)
        MemAlloc::alignedLargePageFree(table);
    size_t allocSize = sizeof(TTBucket) * numBuckets;
    table            = static_cast<TTBucket *>(MemAlloc::alignedLargePageAlloc(allocSize));

    for (size_t i = 0; i < numBuckets; i++) {
        TTBucket &cluster = table[i];
        in->read(reinterpret_cast<char *>(&cluster), sizeof(TTBucket));
    }
    return *in && in->peek() == std::ios::traits_type::eof();
}

int HashTable::hashUsage() const
{
    size_t cnt     = 0;
    size_t testCnt = numBuckets >> 10;

    for (size_t i = 0; i < testCnt; ++i) {
        TTEntry *entry = table[i].entry;
        for (int j = 0; j < ENTRIES_PER_BUCKET; ++j)
            cnt += entry[j].depth8 && entry[j].generation8 == generation;
    }

    return int(cnt * 1000 / (ENTRIES_PER_BUCKET * testCnt));
}

size_t HashTable::hashSizeKB() const
{
    return numBuckets / (1024 / sizeof(TTBucket));
}

}  // namespace Search
