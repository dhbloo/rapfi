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

#include "utils.h"

#define ZOBRISH_SEED 0xa52ca39782739747ULL

namespace Hash {

/// Global zobrist table that will be inititalized at startup.
HashKey zobrist[SIDE_NB][FULL_BOARD_CELL_COUNT];
HashKey zobristSide[SIDE_NB];

/// Init zobrish table using PRNG with the given seed.
/// @param seed Seed of PRNG.
void initZobrish(uint64_t seed)
{
    PRNG prng {seed};

    for (int i = 0; i < FULL_BOARD_CELL_COUNT; i++) {
        zobrist[BLACK][i] = prng();
        zobrist[WHITE][i] = prng();
    }

    zobristSide[BLACK] = prng();
    zobristSide[WHITE] = prng();
}

const auto init = []() {
    initZobrish(ZOBRISH_SEED);
    return true;
}();

}  // namespace Hash
