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
#include <xxhash.h>

namespace Hash {

// Linear congruential hash
constexpr uint64_t LCHash(uint64_t x)
{
    return x * 6364136223846793005ULL + 1442695040888963407ULL;
}

// Hasher based on XXH64
class XXHasher
{
public:
    XXHasher(uint64_t seed = 0) : state(XXH64_createState()) { XXH64_reset(state, seed); }
    XXHasher(const XXHasher &) = delete;
    ~XXHasher() { XXH64_freeState(state); }

    template <typename T>
    XXHasher &operator<<(const T &v)
    {
        XXH64_update(state, static_cast<const void *>(&v), sizeof(T));
        return *this;
    }

    void operator()(const void *input, size_t length) { XXH64_update(state, input, length); }
    template <typename T>
    void operator()(const T *input, size_t count)
    {
        operator()(static_cast<const void *>(input), sizeof(T) * count);
    }

    operator uint64_t() const { return XXH64_digest(state); }

private:
    XXH64_state_t *state;
};

extern HashKey zobrist[SIDE_NB][FULL_BOARD_CELL_COUNT];
extern HashKey zobristSide[SIDE_NB];

}  // namespace Hash
