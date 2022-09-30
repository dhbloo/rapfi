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

/// Pattern2x struct compresses two patterns into one byte to save space.
struct Pattern2x
{
    Pattern patBlack : 4;
    Pattern patWhite : 4;

    Pattern operator[](Color side) const { return side == BLACK ? patBlack : patWhite; }
};
static_assert(sizeof(Pattern2x) == sizeof(uint8_t));

namespace PatternConfig {

template <Rule R>
constexpr int HalfLineLen = R == Rule::FREESTYLE ? 4 : 5;

template <Rule R>
constexpr int KeyLen = HalfLineLen<R> * 4;

template <Rule R>
constexpr int KeyCnt = 1 << KeyLen<R>;

extern Pattern2x   PATTERN2x[KeyCnt<FREESTYLE>];
extern Pattern2x   PATTERN2xStandard[KeyCnt<STANDARD>];
extern Pattern2x   PATTERN2xRenju[KeyCnt<RENJU>];
extern PatternCode PCODE[PATTERN_NB][PATTERN_NB][PATTERN_NB][PATTERN_NB];
extern uint8_t     DEFENCE[KeyCnt<FREESTYLE>][2];
extern uint8_t     DEFENCEStandard[KeyCnt<STANDARD>][2];
extern uint8_t     DEFENCERenju[KeyCnt<RENJU>][2];

/// Remove the center cell in a key according to the rule.
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

/// Lookup line pattern from a 64bit bit key.
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

/// Lookup line pattern from a 64bit bit key.
template <Rule R>
inline uint8_t lookupDefenceTable(uint64_t key, Color attackSide)
{
    static_assert(R == FREESTYLE || R == STANDARD || R == RENJU,
                  "incorrect rule to lookup defence table");
    assert(attackSide == BLACK || attackSide == WHITE);

    key = fuseKey<R>(key);

    //// Bitkey: 00-empty, 01-black, 10-white, 11-wall
    //// Attack should be black, if not, flip all black and white stones
    //// (which is the same operation as reversing all bits in the key)
    // if (attackSide != BLACK) {
    //    if constexpr (HalfLineLen<R> == 5)
    //        key = ((uint64_t)reverseByte(uint8_t(key)) << 12)
    //              | ((uint64_t)reverseByte(uint8_t(key >> 8)) << 4)
    //              | ((uint64_t)reverseByte(uint8_t(key >> 16)) >> 4);  // Reverse 20-bits
    //    else
    //        key = ((uint64_t)reverseByte(uint8_t(key)) << 8)
    //              | (uint64_t)reverseByte(uint8_t(key >> 8));  // Reverse 16-bits
    //}

    if constexpr (R == Rule::FREESTYLE)
        return DEFENCE[key][attackSide];
    else if constexpr (R == Rule::STANDARD)
        return DEFENCEStandard[key][attackSide];
    else if constexpr (R == Rule::RENJU)
        return DEFENCERenju[key][attackSide];
}

}  // namespace PatternConfig
