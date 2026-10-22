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

#include "history.h"

#include "../../game/board.h"
#include "../searchthread.h"
#include "searcher.h"
#include "searchstack.h"

#include <limits>

namespace {

/// History and stats update bonus, based on depth
constexpr int statBonus(Rule rule, Depth d)
{
    constexpr int StatBonusQuadratic[RULE_NB] = {29, 24, 23};
    return std::min(static_cast<int>(StatBonusQuadratic[rule] * d * d + 99 * d - 164), 8927);
}

constexpr int MainHistoryAttackWeight[RULE_NB + 1] = {492, 531, 511, 520};
constexpr int MainHistoryQuietWeight[RULE_NB + 1]  = {265, 270, 270, 268};
constexpr int CounterMoveBonus[RULE_NB + 1]        = {38, 33, 31, 33};

constexpr int historyTableIndex(Rule rule, Color sideToMove)
{
    return int(rule) + (rule == RENJU ? int(sideToMove) : 0);
}

static_assert(std::numeric_limits<int>::max() >= 1LL * 10692 * 2048);

}  // namespace

namespace Search::AB {

MoveHistoryScoring makeMoveHistoryScoring(Rule                      rule,
                                          Color                     sideToMove,
                                          const MainHistory        &mainHistory,
                                          const CounterMoveHistory &counterMoveHistory)
{
    int table = historyTableIndex(rule, sideToMove);
    return {&mainHistory,
            &counterMoveHistory,
            MainHistoryAttackWeight[table],
            MainHistoryQuietWeight[table],
            CounterMoveBonus[table]};
}

void HistoryTracker::addSearchedMove(Pos move, Pos currentBestMove)
{
    if (move == currentBestMove)
        return;

    Color self = board.sideToMove(), oppo = ~self;
    bool  oppo4 = board.p4Count(oppo, A_FIVE) || board.p4Count(oppo, B_FLEX4);

    if (searchStack->moveP4[self] >= H_FLEX3 && attackCount < MAX_ATTACKS)
        attacksSearched[attackCount++] = move;
    else if (!oppo4 && searchStack->moveP4[self] < H_FLEX3 && quietCount < MAX_QUIETS)
        quietsSearched[quietCount++] = move;
}

void HistoryTracker::updateBestmoveStats(Depth depth, Pos bestMove, Value bestValue)
{
    Color    self = board.sideToMove(), oppo = ~self;
    bool     oppo5  = board.p4Count(oppo, A_FIVE);
    bool     oppo4  = oppo5 || board.p4Count(oppo, B_FLEX4);
    Pattern4 selfP4 = board.pattern4(bestMove, self);
    int      bonus  = statBonus(board.thisThread()->options().rule, depth);

    if (selfP4 >= H_FLEX3) {
        searchData->mainHistory[self][bestMove][HIST_ATTACK] << bonus;
    }
    else if (!oppo4 && selfP4 < H_FLEX3) {
        updateQuietStats(bestMove, bonus);

        // Decrease stats for all the other played non-best quiet moves
        for (int i = 0; i < quietCount; i++)
            searchData->mainHistory[self][quietsSearched[i]][HIST_QUIET] << -bonus;
    }

    // Decrease stats for all the other played non-best attack moves
    for (int i = 0; i < attackCount; i++)
        searchData->mainHistory[self][attacksSearched[i]][HIST_ATTACK] << -bonus;

    // Update counter move history if last move is valid (not a pass)
    // Only update if last opponent move is not a four (otherwise we only have one possible reply)
    if (Pos lastMove = board.getLastMove(); !oppo5 && board.isInBoard(lastMove)) {
        searchData->counterMoveHistory[oppo][lastMove.moveIndex()] =
            std::make_pair(bestMove, selfP4);
    }
}

void HistoryTracker::updateTTMoveStats(Depth depth, Pos ttMove, Value ttValue, Value beta)
{
    // Validate ttMove first
    if (!board.isLegal(ttMove))
        return;

    Color    self = board.sideToMove(), oppo = ~self;
    bool     oppo5  = board.p4Count(oppo, A_FIVE);
    bool     oppo4  = oppo5 || board.p4Count(oppo, B_FLEX4);
    Pattern4 selfP4 = board.pattern4(ttMove, self);
    int      bonus  = statBonus(board.thisThread()->options().rule, depth);

    if (!oppo4 && selfP4 < H_FLEX3) {
        // Bonus for a quiet ttMove that fails high
        if (ttValue >= beta)
            updateQuietStats(ttMove, bonus);
        // Penalty for a quiet ttMove that fails low
        else
            searchData->mainHistory[self][ttMove][HIST_QUIET] << -bonus;
    }
}

void HistoryTracker::updateQuietStats(Pos move, int bonus)
{
    Color self = board.sideToMove();

    searchData->mainHistory[self][move][HIST_QUIET] << bonus;
    searchStack->setKiller(move);  // Update killer heuristic move
}

}  // namespace Search::AB
