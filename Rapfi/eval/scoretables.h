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

#include "../core/math.h"
#include "../core/types.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

/// Classical evaluation tables and the engine's value scale.
///
/// This is the query surface of the classical (pattern-based) evaluation: the
/// EVALS/EVALS_THREAT/P4SCORES tables with their inline lookups, and the
/// value <-> winning-rate conversion that defines the meaning of Value across
/// the engine. It is split out of config.h so that hot-path consumers
/// (game/board.h, eval/) do not pull in the whole config surface; the tables
/// are still populated by config.cpp (model loading) and tuning/tuner.cpp
/// (offline tuning).

/// Total count of patterncode (pattern combination for 4 directions)
constexpr uint32_t PCODE_NB = combineNumber(PATTERN_NB, 4);

/// Total count of threat masks (see makeThreatMask() in eval/eval.cpp)
constexpr uint32_t THREAT_NB = power(2, 11);

/// Move-ordering scores of one pattern code: `byOwner` is the score when the color that owns
/// the pattern plays this cell, while `byOpponent` is the score when the other color takes it.
/// Plain int16 fields (no bitfields) so reads are single sign-extending loads; indexed
/// access ([0] owner, [1] opponent) is kept for the config/tuner serialization surface.
struct MoveScorePair
{
    Score byOwner;
    Score byOpponent;

    Score &operator[](size_t idx)
    {
        assert(idx < 2);
        return idx != 0 ? byOpponent : byOwner;
    }
    Score operator[](size_t idx) const
    {
        assert(idx < 2);
        return idx != 0 ? byOpponent : byOwner;
    }
};
static_assert(sizeof(MoveScorePair) == sizeof(int32_t));

namespace Evaluation {

/// Convert a wide move-ordering score to its stored representation. Intermediate arithmetic
/// must remain wide; reaching either limit indicates saturation rather than signed wraparound.
constexpr Score clampMoveScore(int score)
{
    return static_cast<Score>(std::clamp(score,
                                         static_cast<int>(std::numeric_limits<Score>::min()),
                                         static_cast<int>(std::numeric_limits<Score>::max())));
}

static_assert(clampMoveScore(40000) == std::numeric_limits<Score>::max());
static_assert(clampMoveScore(-40000) == std::numeric_limits<Score>::min());

/// Scaling factor of the sigmoid that maps Value to winning rate.
extern float ScalingFactor;

/// Eval and score tables. Renju is asymmetric, so it gets separate black/white
/// tables and the arrays have one extra slot; index them via tableIndex().
/// These hold learned weights, so all four slots are real; the (pcode ->
/// Pattern4) classification is rule logic instead and lives fused inside the
/// PatternConfig::PCODE tables' spare high bits, not here. Unlike the old
/// 14-bit bitfields, stored scores keep full int16 precision (the old
/// narrowing was an artifact of the bundled struct).
extern Eval          EVALS[RULE_NB + 1][PCODE_NB];
extern Eval          EVALS_THREAT[RULE_NB + 1][THREAT_NB];
extern MoveScorePair P4SCORES[RULE_NB + 1][PCODE_NB];

/// Optional low-dimensional post-processing for the freestyle classical value.
/// The readout is an odd monotone piecewise-linear map over normalized value,
/// with fixed x knots {0, 0.5, 1, 2, 4, 8}. The knots live in config.toml so
/// the underlying model artifact remains independently reproducible.
struct ClassicalValueReadout
{
    static constexpr size_t KnotCount = 6;

    bool                          knotsActive = false;
    std::array<double, KnotCount> knots {};
};

extern ClassicalValueReadout CLASSICAL_VALUE_READOUT;

void  refreshClassicalValueReadoutCache();
Value mapClassicalValue(Value rawValue);

inline bool isClassicalValueReadoutActive(Rule rule)
{
    return rule == FREESTYLE && CLASSICAL_VALUE_READOUT.knotsActive;
}

/// Get table index for rule and color.
constexpr int tableIndex(Rule r, Color c)
{
    return r + (r == Rule::RENJU ? c : 0);
}

/// Lookup eval table with color and pcode of rule R.
inline Value getValueBlack(Rule R, PatternCode pcodeBlack, PatternCode pcodeWhite)
{
    Value valueBlack = (Value)EVALS[tableIndex(R, BLACK)][pcodeBlack];
    Value valueWhite = (Value)EVALS[tableIndex(R, WHITE)][pcodeWhite];
    return valueBlack - valueWhite;
}

/// Lookup the move-ordering score table with color and pcode of rule R.
inline MoveScorePair getMoveScorePair(Rule R, Color C, PatternCode pcode)
{
    return P4SCORES[tableIndex(R, C)][pcode];
}

/// Converts a evaluation value to winning rate (in [0, 1]) using current ScalingFactor.
template <bool Strict = true>
inline float valueToWinRate(Value eval)
{
    if (eval >= (Strict ? VALUE_MATE_IN_MAX_PLY : VALUE_EVAL_MAX))
        return 1.0f;
    if (eval <= (Strict ? VALUE_MATED_IN_MAX_PLY : VALUE_EVAL_MIN))
        return 0.0f;
    return 1.0f / (1.0f + ::expf(-float(eval) / ScalingFactor));
}

/// Converts a winning rate in [0, 1] to a evaluation value using current ScalingFactor.
inline Value winRateToValue(float winRate)
{
    float valueF32 = ScalingFactor * ::logf(winRate / (1.0f - winRate));
    valueF32       = std::clamp<float>(valueF32, VALUE_EVAL_MIN, VALUE_EVAL_MAX);
    return Value(valueF32);
}

}  // namespace Evaluation
