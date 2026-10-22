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

void resetPolicyCross()
{
    std::memset(POLICY_CROSS, 0, sizeof(POLICY_CROSS));
    std::memset(POLICY_CROSS_ACTIVE_MASK, 0, sizeof(POLICY_CROSS_ACTIVE_MASK));
}

void installPolicyCross(size_t table, const StoredPolicyCrossTable &values)
{
    assert(table < RULE_NB + 1);
    std::memset(POLICY_CROSS[table], 0, sizeof(POLICY_CROSS[table]));
    uint8_t activeMask = 0;
    size_t  index      = 0;
    for (size_t context = 0; context < PolicyStoredContextCount; context++)
        for (size_t self = 0; self < PolicyStoredPatternCount; self++)
            for (size_t opponent = 0; opponent < PolicyStoredPatternCount; opponent++) {
                const Score score = values[index++];
                POLICY_CROSS[table][context][policyPatternFromStorageIndex(self)]
                            [policyPatternFromStorageIndex(opponent)] = score;
                if (score != 0)
                    activeMask |= uint8_t(1U << context);
            }
    POLICY_CROSS_ACTIVE_MASK[table] = activeMask;
}

StoredPolicyCrossTable storedPolicyCross(size_t table)
{
    assert(table < RULE_NB + 1);
    StoredPolicyCrossTable values;
    size_t                 index = 0;
    for (size_t context = 0; context < PolicyStoredContextCount; context++)
        for (size_t self = 0; self < PolicyStoredPatternCount; self++)
            for (size_t opponent = 0; opponent < PolicyStoredPatternCount; opponent++)
                values[index++] = POLICY_CROSS[table][context][policyPatternFromStorageIndex(self)]
                                              [policyPatternFromStorageIndex(opponent)];
    return values;
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

ClassicalPolicyScorer::ClassicalPolicyScorer(Rule rule, const Board &board)
    : rule_(rule)
    , self_(board.sideToMove())
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

}  // namespace Evaluation
