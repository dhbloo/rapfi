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

#include "../core/pos.h"
#include "scoretables.h"

#include <cstddef>
#include <cstdint>
#include <limits>

class Board;

namespace Evaluation {

/// Tactical contexts used by the compact classical policy residual.
enum PolicyContext : uint8_t {
    OPPONENT_FLEX_FOUR,
    OPPONENT_BLOCK4_FLEX3,
    MUTUAL_RACE,
    SELF_ATTACKING,
    OPPONENT_ATTACKING,
    QUIET,
    POLICY_CONTEXT_NB,
};

constexpr size_t PolicyContextCount       = POLICY_CONTEXT_NB;
constexpr size_t PolicyStoredContextCount = QUIET;
constexpr size_t PolicyStoredPatternCount = PATTERN4_NB - 1;
static_assert(PolicyContextCount <= std::numeric_limits<uint8_t>::digits);
static_assert(PolicyStoredContextCount < PolicyContextCount);
static_assert(FORBID < PATTERN4_NB);

constexpr Pattern4 policyPatternFromStorageIndex(size_t index)
{
    return Pattern4(index + (index >= FORBID));
}

constexpr size_t policyPatternStorageIndex(Pattern4 pattern)
{
    return size_t(pattern) - (pattern > FORBID);
}

extern Score   POLICY_CROSS[RULE_NB + 1][PolicyContextCount][PATTERN4_NB][PATTERN4_NB];
extern uint8_t POLICY_CROSS_ACTIVE_MASK[RULE_NB + 1];
extern bool    POLICY_CROSS_PRESENT;

void resetPolicyCross();
void activatePolicyCross();
void refreshPolicyCrossActivation();
bool hasDefaultPolicyCross();

inline bool isPolicyCrossPresent()
{
    return POLICY_CROSS_PRESENT;
}

inline uint8_t policyCrossActiveMask(Rule rule, Color sideToMove)
{
    return POLICY_CROSS_ACTIVE_MASK[tableIndex(rule, sideToMove)];
}

inline Score
getPolicyCross(Rule rule, Color sideToMove, PolicyContext context, Pattern4 self, Pattern4 opponent)
{
    assert(context < PolicyContextCount);
    return POLICY_CROSS[tableIndex(rule, sideToMove)][context][self][opponent];
}

PolicyContext classifyPolicyContext(const Board &board);

struct ClassicalPolicyBlend
{
    static constexpr int Scale        = 256;
    static constexpr int MaxAbsWeight = 768;

    enum Component {
        SELF_ATTACK,
        SELF_DEFENSE,
        OPPONENT_ATTACK,
        OPPONENT_DEFENSE,
        COMPONENT_NB,
    };
};

inline constexpr int ClassicalPolicyBlendWeights[RULE_NB + 1][ClassicalPolicyBlend::COMPONENT_NB] =
    {
        {235, 264, -25, 121},
        {292, 265, 14, 16},
        {271, 247, 17, 6},
        {289, 272, 16, 40},
};

inline const int *classicalPolicyBlendWeights(Rule rule, Color sideToMove)
{
    return ClassicalPolicyBlendWeights[tableIndex(rule, sideToMove)];
}

constexpr int composeClassicalPolicyScore(int selfAttack,
                                          int selfDefense,
                                          int opponentAttack,
                                          int opponentDefense,
                                          int weightSelfAttack,
                                          int weightSelfDefense,
                                          int weightOpponentAttack,
                                          int weightOpponentDefense)
{
    return (weightSelfAttack * selfAttack + weightSelfDefense * selfDefense
            + weightOpponentAttack * opponentAttack + weightOpponentDefense * opponentDefense)
           / ClassicalPolicyBlend::Scale;
}

inline int composeClassicalPolicyScore(const int *weights,
                                       Score      selfAttack,
                                       Score      selfDefense,
                                       Score      opponentAttack,
                                       Score      opponentDefense)
{
    return composeClassicalPolicyScore(selfAttack,
                                       selfDefense,
                                       opponentAttack,
                                       opponentDefense,
                                       weights[ClassicalPolicyBlend::SELF_ATTACK],
                                       weights[ClassicalPolicyBlend::SELF_DEFENSE],
                                       weights[ClassicalPolicyBlend::OPPONENT_ATTACK],
                                       weights[ClassicalPolicyBlend::OPPONENT_DEFENSE]);
}

int classicalPolicyBaseScore(Rule rule, Color self, PatternCode pcodeBlack, PatternCode pcodeWhite);

/// Position-bound classical policy scorer. Context and blend selection are computed once,
/// while score() remains an inline per-candidate hot-path operation.
class ClassicalPolicyScorer
{
public:
    ClassicalPolicyScorer(Rule rule, const Board &board);

    Score score(PatternCode pcodeBlack,
                PatternCode pcodeWhite,
                Pattern4    selfPattern,
                Pattern4    opponentPattern) const
    {
        MoveScorePair patternScores[SIDE_NB] = {
            getMoveScorePair(rule_, BLACK, pcodeBlack),
            getMoveScorePair(rule_, WHITE, pcodeWhite),
        };
        const Score selfAttack      = patternScores[self_].byOwner;
        const Score selfDefense     = patternScores[opponent_].byOpponent;
        const Score opponentAttack  = patternScores[opponent_].byOwner;
        const Score opponentDefense = patternScores[self_].byOpponent;
        int result = composeClassicalPolicyScore(
            blendWeights_, selfAttack, selfDefense, opponentAttack, opponentDefense);
        if (policyCrossActive_)
            result += getPolicyCross(rule_, self_, context_, selfPattern, opponentPattern);
        return clampMoveScore(result);
    }

    Score score(const Board &board, Pos pos) const;

private:
    Rule          rule_;
    Color         self_;
    Color         opponent_;
    const int    *blendWeights_;
    PolicyContext context_;
    bool          policyCrossActive_;
};

static_assert(std::numeric_limits<int>::max() >= 4LL * (1LL << std::numeric_limits<Score>::digits)
                                                     * ClassicalPolicyBlend::MaxAbsWeight);
static_assert(composeClassicalPolicyScore(123, -45, 77, -91, 256, 256, 0, 0) == 78);
static_assert(composeClassicalPolicyScore(-123, 45, -77, 91, 256, 256, 0, 0) == -78);
static_assert(composeClassicalPolicyScore(-123, 0, 0, 0, 255, 0, 0, 0) == -122);

}  // namespace Evaluation
