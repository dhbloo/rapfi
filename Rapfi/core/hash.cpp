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

#include "hash.h"

#include "random.h"  // PRNG (the only utility hash.cpp needs)

namespace Hash {

/// Seed for the Zobrist table PRNG. Changing this value invalidates all transposition table
/// contents persisted between runs and is part of what makes the bench hash deterministic.
constexpr uint64_t ZOBRIST_SEED = 0xa52ca39782739747ULL;

HashKey zobrist[SIDE_NB][FULL_BOARD_CELL_COUNT];
HashKey zobristSide[SIDE_NB];

/// Fill the Zobrist tables with a deterministic stream of pseudo-random keys.
static void initZobrist(uint64_t seed)
{
    PRNG prng {seed};

    for (int i = 0; i < FULL_BOARD_CELL_COUNT; i++) {
        zobrist[BLACK][i] = prng();
        zobrist[WHITE][i] = prng();
    }

    zobristSide[BLACK] = prng();
    zobristSide[WHITE] = prng();
}

/// Run `initZobrist` once at program start, before any board is constructed.
[[maybe_unused]] static const bool zobristInitialised = [] {
    initZobrist(ZOBRIST_SEED);
    return true;
}();

}  // namespace Hash
