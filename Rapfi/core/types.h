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
// Fundamental scalar aliases used across the engine

using PatternCode = uint16_t;  ///< Packed key into the four-direction pattern combination table.
using Score       = int16_t;   ///< Move-ordering / pattern weight, additive.
using Eval        = int16_t;   ///< Static evaluation from the classical evaluator.
using Depth       = float;     ///< Search depth; fractional, negative for VCF / QVCF.
using HashKey     = uint64_t;  ///< 64-bit Zobrist key for a board position.

// -------------------------------------------------
// Alpha-beta bounds and depth constants

/// Type of value bound stored in a TT entry.
enum Bound {
    BOUND_NONE,
    BOUND_UPPER,                              ///< Fail-low bound (value <= alpha).
    BOUND_LOWER,                              ///< Fail-high bound (value >= beta).
    BOUND_EXACT = BOUND_UPPER | BOUND_LOWER,  ///< PV-node exact value.
};

constexpr Depth DEPTH_QVCF_FULL = -1.0f;  ///< Quiescent VCF including defender replies.
constexpr Depth DEPTH_QVCF      = -2.0f;  ///< Quiescent VCF, attacker side only.
constexpr Depth DEPTH_NONE      = -3.0f;  ///< Sentinel for "no depth recorded".

/// Depths are stored in TT entries relative to `DEPTH_LOWER_BOUND`; the resulting range is
/// 256 plies wide, hence `DEPTH_UPPER_BOUND = DEPTH_LOWER_BOUND + 255`.
constexpr Depth DEPTH_LOWER_BOUND = -20.0f;
constexpr Depth DEPTH_UPPER_BOUND = DEPTH_LOWER_BOUND + 255;

// -------------------------------------------------
// Stone color

/// Color of a stone (or non-stone state) on a board cell.
///
/// The numeric layout is load-bearing: BLACK=0, WHITE=1, WALL=2, EMPTY=3 lets `operator~`
/// flip between BLACK<->WHITE and WALL<->EMPTY with a single XOR.
enum Color : uint8_t {
    BLACK,
    WHITE,
    WALL,
    EMPTY,
    COLOR_NB,     ///< Total number of distinct cell states.
    SIDE_NB = 2,  ///< Number of playable sides (BLACK and WHITE).
};

/// Returns the opposite color: BLACK<->WHITE, WALL<->EMPTY.
constexpr Color operator~(Color p)
{
    return Color(p ^ 1);
}

// -------------------------------------------------
// Line patterns and four-direction pattern combinations
//
// Diagram notation used in the per-value docs below:
//   O = our stone, X = opponent stone, _ = empty cell, . = the centre cell being classified.

/// Single-direction line pattern at one cell, ordered roughly by threat strength.
enum Pattern : uint8_t {
    DEAD,  ///< X_.__X  - blocked on both sides, no five possible.
    OL,    ///< OO.OOO  - one step before an overline (matters under STANDARD/RENJU).
    B1,    ///< X.____  - one step before B2.
    F1,    ///< X_.___  - one step before F2.
    B2,    ///< XO.___  - one step before B3.
    F2,    ///< _O__._  - one step before two F3.
    F2A,   ///< _O_.__  - one step before three F3.
    F2B,   ///< _O.___  - one step before four F3.
    B3,    ///< XOO.__  - one step before B4.
    F3,    ///< _OO_._  - one step before one F4.
    F3S,   ///< __OO.__ - one step before two F4.
    B4,    ///< XOOO._X - one step before F5.
    F4,    ///< _OOO._X - one step before two F5.
    F5,    ///< XOO.OOX - completes a five (wins, except overlines under STANDARD/RENJU).
    PATTERN_NB
};

/// Aggregate pattern combining the four line directions at a cell, ordered from weak (NONE)
/// to mate (A_FIVE). Derived from four `Pattern` values via `PatternConfig::PCODE`.
enum Pattern4 : uint8_t {
    NONE,            ///< Nothing notable.
    FORBID,          ///< Renju forbidden point for the side to move.
    L_FLEX2,         ///< F2 + Any.
    K_BLOCK3,        ///< B3 + Any.
    J_FLEX2_2X,      ///< F2 x 2.
    I_BLOCK3_PLUS,   ///< B3 x 2  or  B3 + F2.
    H_FLEX3,         ///< F3 + Any.
    G_FLEX3_PLUS,    ///< F3 + F2  or  F3 + B3.
    F_FLEX3_2X,      ///< F3 x 2.
    E_BLOCK4,        ///< B4 + Any.
    D_BLOCK4_PLUS,   ///< B4 + F2  or  B4 + B3.
    C_BLOCK4_FLEX3,  ///< B4 + F3.
    B_FLEX4,         ///< F4  or  F4S  or  B4 x 2.
    A_FIVE,          ///< F5 - immediate win.
    PATTERN4_NB
};

// -------------------------------------------------
// Search value with mate-distance encoding

/// Search result value, encoding both normal evaluations and mate distances in one integer.
/// Positive = side to move is winning. Magnitudes near `VALUE_MATE` represent mate distance
/// from the root; see `mate_in` / `mated_in` and the TT-storage conversion helpers below.
enum Value {
    VALUE_ZERO     = 0,
    VALUE_DRAW     = 0,
    VALUE_MATE     = 30000,
    VALUE_INFINITE = 30001,
    VALUE_NONE     = -30002,  ///< Sentinel for "no value yet" / TT miss.
    VALUE_BLOCKED  = -30003,  ///< Sentinel for a position with no legal candidate move.

    /// Search-side mate values are clamped to these bounds, leaving 500 plies of headroom for
    /// the longest representable mate distance.
    VALUE_MATE_IN_MAX_PLY  = VALUE_MATE - 500,
    VALUE_MATED_IN_MAX_PLY = -VALUE_MATE + 500,

    /// Distinguishes mate values that came from the database (treated as the longest possible
    /// mate so that a shorter, search-derived mate always preempts them).
    VALUE_MATE_FROM_DATABASE  = VALUE_MATE_IN_MAX_PLY,
    VALUE_MATED_FROM_DATABASE = VALUE_MATED_IN_MAX_PLY,

    VALUE_EVAL_MAX = 6000,   ///< Upper clamp for non-mate evaluations.
    VALUE_EVAL_MIN = -6000,  ///< Lower clamp for non-mate evaluations.
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

/// Value representing "side to move mates in `ply` plies".
constexpr Value mate_in(int ply)
{
    return Value(int(VALUE_MATE) - ply);
}

/// Value representing "side to move is mated in `ply` plies".
constexpr Value mated_in(int ply)
{
    return Value(int(-VALUE_MATE) + ply);
}

/// Number of plies until mate, given a (mate-class) Value and the search ply at which it was
/// produced. Result is undefined for non-mate values.
constexpr int mate_step(Value v, int ply)
{
    return VALUE_MATE - ply - (v < 0 ? -v : v);
}

/// Convert a search value into the form stored in a TT entry.
///
/// During search, mate values are encoded relative to the *root*: `VALUE_MATE - 7` means
/// "mate in 7 plies from the root". The TT instead stores mate values relative to the
/// *position where the entry was written*, so a single entry stays correct regardless of the
/// path that reaches it. `searchPly` is the ply at which `value` was produced.
constexpr int searchValueToStoredValue(Value value, int searchPly)
{
    return value == VALUE_NONE               ? VALUE_NONE
           : value >= VALUE_MATE_IN_MAX_PLY  ? value + searchPly
           : value <= VALUE_MATED_IN_MAX_PLY ? value - searchPly
                                             : value;
}

/// Inverse of `searchValueToStoredValue`: convert a TT-stored value back into one comparable
/// at the current search ply.
constexpr Value storedValueToSearchValue(int storedValue, int searchPly)
{
    return storedValue == VALUE_NONE               ? VALUE_NONE
           : storedValue >= VALUE_MATE_IN_MAX_PLY  ? Value(storedValue - searchPly)
           : storedValue <= VALUE_MATED_IN_MAX_PLY ? Value(storedValue + searchPly)
                                                   : Value(storedValue);
}

// -------------------------------------------------
// Game rule and search-action types

/// Five-in-a-row variant: decides what counts as a win and which moves are forbidden.
enum Rule : uint8_t {
    FREESTYLE,  ///< Free Gomoku: five or more in a row wins; no other restrictions.
    STANDARD,   ///< Standard Gomoku: exactly five wins (overlines do not).
    RENJU,      ///< Renju: black is forbidden from 3-3, 4-4, and overlines; white unrestricted.
    RULE_NB
};

/// Full game configuration: a `Rule` plus an opening protocol.
struct GameRule
{
    Rule rule;

    /// Opening protocol used at the start of the game. Implemented in `search/opening.{h,cpp}`.
    enum OpeningRule : uint8_t {
        FREEOPEN,  ///< No opening protocol; players simply alternate moves from move 1.
        SWAP1,     ///< One-side swap.
        SWAP2,     ///< Standard swap2 (used in Gomocup tournaments).
    } opRule;

    operator Rule() const { return rule; }
};

/// Action returned by the search engine to its caller.
enum class ActionType {
    Move,         ///< Play one move at the returned position.
    Move2,        ///< Play two moves (balance-2 mode).
    Swap,         ///< Swap colors (swap protocol).
    Swap2PutTwo,  ///< Under SWAP2, decline both options and place two additional stones.
};
