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

#include "../game/movegen.h"
#include "history.h"

namespace Search {

/// MovePicker class is used to pick one legal move at a time from the current
/// position. In order to improve the efficiency of the alpha beta algorithm,
/// it attempts to return the moves which are most likely to get a cut-off first.
class MovePicker
{
public:
    enum SearchType {
        ROOT,
        MAIN,
        QVCF,
    };
    template <SearchType ST>
    struct ExtraArgs
    {};

    /// Construct a MovePicker object with arguments used to decide which
    /// generation phase to set and provide information for move ordering.
    template <SearchType ST>
    MovePicker(Rule rule, const Board &board, ExtraArgs<ST> args);
    MovePicker(const MovePicker &)            = delete;
    MovePicker &operator=(const MovePicker &) = delete;

    /// Gets the next move in the sorted move list.
    [[nodiscard]] Pos operator()();

    /// Whether this movepicker has policy score from the evaluator.
    bool hasPolicyScore() const { return hasPolicy; }
    /// Whether this movepicker computes normalized policy score.
    bool hasNormalizedPolicy() const { return useNormalizedPolicy; }
    /// Get the normalized policy in [0,1] of current move. Use after enable normalized policy.
    float curMoveNormalizePolicy() const { return curPolicy; }
    /// Get the maximum policy score of all moves.
    Score maxMovePolicy() const { return maxPolicyScore; }
    /// Get the policy score of current move.
    Score curMovePolicy() const { return curPolicyScore; }
    /// Get adjusted score of current move.
    Score curMoveScore() const { return curScore; }
    /// Get the policy score difference between current move and the best move.
    Score curMovePolicyDiff() const { return maxPolicyScore - curPolicyScore; }
    /// Get the adjusted score difference between current move and the best move.
    Score curMoveScoreDiff() const { return maxScore - curScore; }

private:
    enum PickType { Next, Best };
    enum ScoreType {
        ATTACK       = 0b01,
        DEFEND       = 0b10,
        BALANCED     = ATTACK | DEFEND,
        POLICY       = 0b100,
        MAIN_HISTORY = 0b1000,
        COUNTER_MOVE = 0b10000,
    };

    template <PickType T, typename Pred>
    Pos pickNextMove(Pred);
    template <ScoreType T>
    void        scoreAllMoves();
    ScoredMove *begin() { return curMove; }
    ScoredMove *end() { return endMove; }

    const Board              &board;
    const MainHistory        *mainHistory;
    const CounterMoveHistory *counterMoveHistory;
    int8_t                    stage;
    Rule                      rule;
    Pos                       ttMove;
    bool                      allowPlainB4InVCF;
    bool                      hasPolicy;
    bool                      useNormalizedPolicy;
    float                     normalizedPolicyTemp;
    Score                     curScore, maxScore;
    Score                     curPolicyScore, maxPolicyScore;
    float                     curPolicy;
    ScoredMove               *curMove, *endMove;
    ScoredMove                moves[MAX_MOVES];
};

template <>
struct MovePicker::ExtraArgs<MovePicker::ROOT>
{
    bool  useNormalizedPolicy  = false;
    float normalizedPolicyTemp = 1.0f;
};

template <>
struct MovePicker::ExtraArgs<MovePicker::MAIN>
{
    Pos                 ttMove;
    MainHistory        *mainHistory;
    CounterMoveHistory *counterMoveHistory;
    bool                useNormalizedPolicy  = false;
    float               normalizedPolicyTemp = 1.0f;
};

template <>
struct MovePicker::ExtraArgs<MovePicker::QVCF>
{
    Pos      ttMove;
    Depth    depth;  // negative depth in qvcf search
    Pattern4 previousSelfP4[2];
};

}  // namespace Search
