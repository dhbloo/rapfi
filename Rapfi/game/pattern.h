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
#include "../core/types.h"

#include <cassert>

/// Packs the line patterns of both colors for one cell into a single byte (4 bits each), so a
/// `Cell` can store all four directions compactly.
struct Pattern2x
{
    Pattern patBlack : 4;  ///< Line pattern seen from black's perspective.
    Pattern patWhite : 4;  ///< Line pattern seen from white's perspective.

    Pattern operator[](Color side) const { return side == BLACK ? patBlack : patWhite; }
};
static_assert(sizeof(Pattern2x) == sizeof(uint8_t));

namespace PatternConfig {

/// Number of cells on each side of the center cell in a line. Freestyle only cares about five in
/// a row (4 neighbours suffice); standard and renju must also detect overlines, needing 5.
template <Rule R>
constexpr int HalfLineLen = R == Rule::FREESTYLE ? 4 : 5;

/// Bit width of a fused line key: two half-lines (the center cell is implicit), 2 bits per cell.
template <Rule R>
constexpr int KeyLen = HalfLineLen<R> * 4;

/// Number of distinct fused line keys, i.e. the size of every per-rule lookup table.
template <Rule R>
constexpr int KeyCnt = 1 << KeyLen<R>;

/// Fused line key -> line pattern of both colors, one table per rule.
extern Pattern2x PATTERN2x[KeyCnt<FREESTYLE>];
extern Pattern2x PATTERN2xStandard[KeyCnt<STANDARD>];
extern Pattern2x PATTERN2xRenju[KeyCnt<RENJU>];
/// The four line patterns around a cell (one per direction) -> a single packed pattern code,
/// order-independent so the four directions can be supplied in any order.
extern PatternCode PCODE[PATTERN_NB][PATTERN_NB][PATTERN_NB][PATTERN_NB];
/// Fused line key + attacking color -> bitmask of cells that defend against the attack.
extern uint8_t DEFENCE[KeyCnt<FREESTYLE>][2];
extern uint8_t DEFENCEStandard[KeyCnt<STANDARD>][2];
extern uint8_t DEFENCERenju[KeyCnt<RENJU>][2];

/// Drop the (implicit, always-self) center cell from a raw line key, yielding the fused key used
/// to index the lookup tables. BMI2's parallel-extract collapses the two half-lines in one op.
template <Rule R>
inline uint64_t fuseKey(uint64_t key)
{
    static_assert(HalfLineLen<R> == 4 || HalfLineLen<R> == 5, "incorrect half line length");

    if constexpr (HalfLineLen<R> == 4) {
#ifdef USE_BMI2
        key = _pext_u64(key, 0x3fcff);
#else
        key = ((key >> 2) & 0xff00) | (key & 0x00ff);
#endif
    }
    else if constexpr (HalfLineLen<R> == 5) {
#ifdef USE_BMI2
        key = _pext_u64(key, 0x3ff3ff);
#else
        key = ((key >> 2) & 0xffc00) | (key & 0x003ff);
#endif
    }

    return key;
}

/// Look up the line pattern of both colors from a raw 64-bit line key.
template <Rule R>
inline Pattern2x lookupPattern(uint64_t key)
{
    static_assert(R == FREESTYLE || R == STANDARD || R == RENJU,
                  "incorrect rule to lookup pattern");

    key = fuseKey<R>(key);

    if constexpr (R == Rule::FREESTYLE)
        return PATTERN2x[key];
    else if constexpr (R == Rule::STANDARD)
        return PATTERN2xStandard[key];
    else if constexpr (R == Rule::RENJU)
        return PATTERN2xRenju[key];
}

/// Look up the defence-move bitmask for `attackSide` from a raw 64-bit line key. The bitkey
/// encodes each cell in 2 bits (00 wall, 01 white, 10 black, 11 empty); the table carries a
/// separate entry per attacking color rather than normalizing the key to a single perspective.
template <Rule R>
inline uint8_t lookupDefenceTable(uint64_t key, Color attackSide)
{
    static_assert(R == FREESTYLE || R == STANDARD || R == RENJU,
                  "incorrect rule to lookup defence table");
    assert(attackSide == BLACK || attackSide == WHITE);

    key = fuseKey<R>(key);

    if constexpr (R == Rule::FREESTYLE)
        return DEFENCE[key][attackSide];
    else if constexpr (R == Rule::STANDARD)
        return DEFENCEStandard[key][attackSide];
    else if constexpr (R == Rule::RENJU)
        return DEFENCERenju[key][attackSide];
}

}  // namespace PatternConfig
