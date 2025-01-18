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

#include <climits>
#include <cstdint>

// -------------------------------------------------
// Common types

typedef uint16_t PatternCode;
typedef int16_t  Score;
typedef int16_t  Eval;
typedef float    Depth;
typedef uint64_t HashKey;

// -------------------------------------------------
// Range of searching depth and bound

/// Type of value bound of the alpha-beta search
enum Bound {
    BOUND_NONE,
    BOUND_UPPER,                              /// Alpha bound
    BOUND_LOWER,                              /// Beta bound
    BOUND_EXACT = BOUND_UPPER | BOUND_LOWER,  /// PV bound
};

constexpr Depth DEPTH_QVCF_FULL   = -1.0f;
constexpr Depth DEPTH_QVCF        = -2.0f;
constexpr Depth DEPTH_NONE        = -3.0f;
constexpr Depth DEPTH_LOWER_BOUND = -20.0f;
constexpr Depth DEPTH_UPPER_BOUND = DEPTH_LOWER_BOUND + 255;

// -------------------------------------------------

/// Color represents the type of piece on board
enum Color : uint8_t {
    BLACK,
    WHITE,
    WALL,
    EMPTY,
    COLOR_NB,     // Total number of color on board
    SIDE_NB = 2,  // Two side of stones (Black and White)
};

// Returns the opposite of a color (Black <-> White, Wall <-> Empty)
constexpr Color operator~(Color p)
{
    return Color(p ^ 1);
}

// -------------------------------------------------

/// Pattern is the type of a single line at one cell
enum Pattern : uint8_t {
    DEAD,  // X_.__X, can never make a five
    OL,    // OO.OOO, one step before overline
    B1,    // X.____, one step before B2
    F1,    // X_.___, one step before F2
    B2,    // XO.___, one step before B3
    F2,    // _O__._, one step before two F3
    F2A,   // _O_.__, one step before three F3
    F2B,   // _O.___, one step before four F3
    B3,    // XOO.__, one step before B4
    F3,    // _OO_._, one step before one F4
    F3S,   // __OO.__, one step before two F4
    B4,    // XOOO._X, one step before F5
    F4,    // _OOO._X, one step before two F5
    F5,    // XOO.OOX, making a five
    PATTERN_NB
};

/// Pattern4 is the combined type of 4 lines at one cell
enum Pattern4 : uint8_t {
    NONE,            // Anything else
    FORBID,          // Forbidden point (for renju)
    L_FLEX2,         // F2+Any
    K_BLOCK3,        // B3+Any
    J_FLEX2_2X,      // F2x2
    I_BLOCK3_PLUS,   // B3x2 | B3+F2
    H_FLEX3,         // F3+Any
    G_FLEX3_PLUS,    // F3+F2 | F3+B3
    F_FLEX3_2X,      // F3x2
    E_BLOCK4,        // B4+Any
    D_BLOCK4_PLUS,   // B4+F2 | B4+B3
    C_BLOCK4_FLEX3,  // B4+F3
    B_FLEX4,         // F4 | F4S | B4x2
    A_FIVE,          // F5
    PATTERN4_NB
};

// -------------------------------------------------

// Integer value that representing the result of a search
enum Value {
    VALUE_ZERO     = 0,
    VALUE_DRAW     = 0,
    VALUE_MATE     = 30000,
    VALUE_INFINITE = 30001,
    VALUE_NONE     = -30002,
    VALUE_BLOCKED  = -30003,

    VALUE_MATE_IN_MAX_PLY     = VALUE_MATE - 500,
    VALUE_MATED_IN_MAX_PLY    = -VALUE_MATE + 500,
    VALUE_MATE_FROM_DATABASE  = VALUE_MATE_IN_MAX_PLY,
    VALUE_MATED_FROM_DATABASE = VALUE_MATED_IN_MAX_PLY,

    VALUE_EVAL_MAX = 6000,
    VALUE_EVAL_MIN = -6000,
};

constexpr Value operator+(Value d1, Value d2)
{
    return Value(int(d1) + int(d2));
}
constexpr Value operator-(Value d1, Value d2)
{
    return Value(int(d1) - int(d2));
}
constexpr Value operator-(Value d)
{
    return Value(-int(d));
}
inline Value &operator+=(Value &d1, Value d2)
{
    return d1 = d1 + d2;
}
inline Value &operator-=(Value &d1, Value d2)
{
    return d1 = d1 - d2;
}
constexpr Value operator*(int i, Value d)
{
    return Value(i * int(d));
}
constexpr Value operator*(Value d, int i)
{
    return Value(int(d) * i);
}
constexpr Value operator/(Value d, int i)
{
    return Value(int(d) / i);
}
constexpr int operator/(Value d1, Value d2)
{
    return int(d1) / int(d2);
}
inline Value &operator*=(Value &d, int i)
{
    return d = Value(int(d) * i);
}
inline Value &operator/=(Value &d, int i)
{
    return d = Value(int(d) / i);
}
constexpr Value operator+(Value v, int i)
{
    return Value(int(v) + i);
}
constexpr Value operator-(Value v, int i)
{
    return Value(int(v) - i);
}
inline Value &operator+=(Value &v, int i)
{
    return v = v + i;
}
inline Value &operator-=(Value &v, int i)
{
    return v = v - i;
}

/// Construct a value for mate in N ply
constexpr Value mate_in(int ply)
{
    return Value(int(VALUE_MATE) - ply);
}

/// Construct a value for being mated in N ply
constexpr Value mated_in(int ply)
{
    return Value(int(-VALUE_MATE) + ply);
}

/// Get number of steps to mate from value and current ply
constexpr int mate_step(Value v, int ply)
{
    return VALUE_MATE - ply - (v < 0 ? -v : v);
}

constexpr int searchValueToStoredValue(Value value, int searchPly)
{
    return value == VALUE_NONE               ? VALUE_NONE
           : value >= VALUE_MATE_IN_MAX_PLY  ? value + searchPly
           : value <= VALUE_MATED_IN_MAX_PLY ? value - searchPly
                                             : value;
}

constexpr Value storedValueToSearchValue(int storedValue, int searchPly)
{
    return storedValue == VALUE_NONE               ? VALUE_NONE
           : storedValue >= VALUE_MATE_IN_MAX_PLY  ? Value(storedValue - searchPly)
           : storedValue <= VALUE_MATED_IN_MAX_PLY ? Value(storedValue + searchPly)
                                                   : Value(storedValue);
}

// -------------------------------------------------

/// Rule is the fundamental rule of the game
enum Rule : uint8_t { FREESTYLE, STANDARD, RENJU, RULE_NB };

/// GameRule is composed of Rule and OpeningRule
struct GameRule
{
    Rule rule;
    enum OpeningRule : uint8_t { FREEOPEN, SWAP1, SWAP2 } opRule;

    operator Rule() const { return rule; }
};

/// Thinking result action type
enum class ActionType {
    Move,
    Move2,
    Swap,
    Swap2PutTwo,
};

// -------------------------------------------------

/// CandidateRange represents the options for different range of move condidates.
enum class CandidateRange {
    SQUARE2,
    SQUARE2_LINE3,
    SQUARE3,
    SQUARE3_LINE4,
    SQUARE4,
    FULL_BOARD,
    CAND_RANGE_NB,
};
