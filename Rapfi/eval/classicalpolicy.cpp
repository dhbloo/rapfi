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

#include "classicalpolicy.h"

#include "../game/board.h"

#include <cstring>

namespace Evaluation {

Score   POLICY_CROSS[RULE_NB + 1][PolicyContextCount][PATTERN4_NB][PATTERN4_NB];
uint8_t POLICY_CROSS_ACTIVE_MASK[RULE_NB + 1] = {};
bool    POLICY_CROSS_PRESENT                  = false;

void resetPolicyCross()
{
    std::memset(POLICY_CROSS, 0, sizeof(POLICY_CROSS));
    std::memset(POLICY_CROSS_ACTIVE_MASK, 0, sizeof(POLICY_CROSS_ACTIVE_MASK));
    POLICY_CROSS_PRESENT = false;
}

void activatePolicyCross()
{
    POLICY_CROSS_PRESENT = true;
    refreshPolicyCrossActivation();
}

void refreshPolicyCrossActivation()
{
    std::memset(POLICY_CROSS_ACTIVE_MASK, 0, sizeof(POLICY_CROSS_ACTIVE_MASK));
    for (size_t table = 0; table < RULE_NB + 1; table++)
        for (size_t context = 0; context < PolicyStoredContextCount; context++)
            for (const auto &row : POLICY_CROSS[table][context])
                for (Score score : row)
                    if (score != 0)
                        POLICY_CROSS_ACTIVE_MASK[table] |= uint8_t(1U << context);
}

bool hasDefaultPolicyCross()
{
    for (const auto &table : POLICY_CROSS)
        for (const auto &context : table)
            for (const auto &row : context)
                for (Score score : row)
                    if (score != 0)
                        return false;
    return true;
}

PolicyContext classifyPolicyContext(const Board &board)
{
    Color self = board.sideToMove(), opponent = ~self;

    if (board.p4Count(opponent, B_FLEX4))
        return OPPONENT_FLEX_FOUR;
    if (board.p4Count(opponent, C_BLOCK4_FLEX3))
        return OPPONENT_BLOCK4_FLEX3;

    bool selfFlex3OrStronger     = false;
    bool opponentFlex3OrStronger = false;
    for (Pattern4 pattern = H_FLEX3; pattern < PATTERN4_NB; pattern = Pattern4(pattern + 1)) {
        selfFlex3OrStronger |= board.p4Count(self, pattern) != 0;
        opponentFlex3OrStronger |= board.p4Count(opponent, pattern) != 0;
    }

    if (selfFlex3OrStronger && opponentFlex3OrStronger)
        return MUTUAL_RACE;
    if (selfFlex3OrStronger)
        return SELF_ATTACKING;
    if (opponentFlex3OrStronger)
        return OPPONENT_ATTACKING;
    return QUIET;
}

int classicalPolicyBaseScore(Rule rule, Color self, PatternCode pcodeBlack, PatternCode pcodeWhite)
{
    Color         opponent               = ~self;
    MoveScorePair patternScores[SIDE_NB] = {
        getMoveScorePair(rule, BLACK, pcodeBlack),
        getMoveScorePair(rule, WHITE, pcodeWhite),
    };
    const Score selfAttack      = patternScores[self].byOwner;
    const Score selfDefense     = patternScores[opponent].byOpponent;
    const Score opponentAttack  = patternScores[opponent].byOwner;
    const Score opponentDefense = patternScores[self].byOpponent;
    const int  *weights         = classicalPolicyBlendWeights(rule, self);
    return composeClassicalPolicyScore(
        weights, selfAttack, selfDefense, opponentAttack, opponentDefense);
}

ClassicalPolicyScorer::ClassicalPolicyScorer(Rule rule, const Board &board)
    : rule_(rule)
    , self_(board.sideToMove())
    , opponent_(~self_)
    , blendWeights_(classicalPolicyBlendWeights(rule, self_))
    , context_(QUIET)
    , policyCrossActive_(false)
{
    uint8_t policyCrossMask = policyCrossActiveMask(rule_, self_);
    if (policyCrossMask) {
        context_           = classifyPolicyContext(board);
        policyCrossActive_ = policyCrossMask & uint8_t(1U << context_);
    }
}

Score ClassicalPolicyScorer::score(const Board &board, Pos pos) const
{
    const auto [pcodeBlack, pcodeWhite] = board.pcodePair(pos);
    return score(pcodeBlack,
                 pcodeWhite,
                 board.pattern4(pos, self_),
                 board.pattern4(pos, opponent_));
}

}  // namespace Evaluation
