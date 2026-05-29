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

#include "pos.h"
#include "types.h"

#include <cstdint>
#include <type_traits>
#include <xxhash.h>

namespace Hash {

/// Linear congruential mix of a single 64-bit value. Cheap, deterministic, and just well-mixed
/// enough for combining a position key with an exclusion mask (see ab/search.cpp).
constexpr uint64_t LCHash(uint64_t x)
{
    return x * 6364136223846793005ULL + 1442695040888963407ULL;
}

/// Incremental hasher built on the XXH64 streaming API. Used by the bench to fingerprint a
/// completed search and by the data writer to identify duplicate self-play games.
class XXHasher
{
public:
    XXHasher(uint64_t seed = 0) : state(XXH64_createState()) { XXH64_reset(state, seed); }
    XXHasher(const XXHasher &)            = delete;
    XXHasher(XXHasher &&)                 = delete;
    XXHasher &operator=(const XXHasher &) = delete;
    XXHasher &operator=(XXHasher &&)      = delete;
    ~XXHasher() { XXH64_freeState(state); }

    /// Feed a trivially-copyable value into the hash. Hashing the object representation of a
    /// type with padding would mix in indeterminate bytes, so this is restricted to trivially-
    /// copyable types (still no guarantee of being padding-free, but it rules out the worst).
    template <typename T>
    XXHasher &operator<<(const T &v)
    {
        static_assert(std::is_trivially_copyable_v<T>, "XXHasher hashes the object representation");
        XXH64_update(state, static_cast<const void *>(&v), sizeof(T));
        return *this;
    }

    /// Feed `length` bytes from `input` into the hash.
    void operator()(const void *input, size_t length) { XXH64_update(state, input, length); }

    /// Feed `count` trivially-copyable values from `input` into the hash.
    template <typename T>
    void operator()(const T *input, size_t count)
    {
        static_assert(std::is_trivially_copyable_v<T>, "XXHasher hashes the object representation");
        operator()(static_cast<const void *>(input), sizeof(T) * count);
    }

    /// Snapshot the current digest. The hasher remains usable after this call.
    operator uint64_t() const { return XXH64_digest(state); }

private:
    XXH64_state_t *state;
};

/// Per-(color, addressable cell) Zobrist key; XOR these in/out of the board hash as stones
/// are placed and removed. Initialised once at program start from a fixed seed so engine
/// instances agree on transposition table contents.
extern HashKey zobrist[SIDE_NB][FULL_BOARD_CELL_COUNT];

/// Per-side-to-move Zobrist key; XOR'd into the board hash when the side to move flips.
extern HashKey zobristSide[SIDE_NB];

}  // namespace Hash
