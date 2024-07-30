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

namespace {

/// History and stats update bonus, based on depth
constexpr int statBonus(Depth d)
{
    return std::min(static_cast<int>(25 * d * d + 105 * d - 157), 8927);
}

}  // namespace

namespace Search::AB {

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
    Pattern4 selfP4 = board.cell(bestMove).pattern4[self];
    int      bonus  = statBonus(depth);

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
    Pattern4 selfP4 = board.cell(ttMove).pattern4[self];
    int      bonus  = statBonus(depth);

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
    searchStack->setKiller(move);  // Update killer heruistic move
}

}  // namespace Search::AB
