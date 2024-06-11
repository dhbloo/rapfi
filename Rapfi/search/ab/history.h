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

#include "../../game/board.h"
#include "../history.h"
#include "../searchthread.h"

namespace Search::AB {

struct ABSearchData;  // forward declaration
struct SearchStack;   // forward declaration

/// HistoryTracker is used to record all information needed to update
/// move heruistics in one search ply in ABSearch.
struct HistoryTracker
{
    static constexpr int MAX_ATTACKS = 48;
    static constexpr int MAX_QUIETS  = 192;

    HistoryTracker(const Board &board, SearchStack *ss)
        : board(board)
        , searchData(board.thisThread()->searchDataAs<ABSearchData>())
        , searchStack(ss)
    {}

    /// Add a searched move to heruistic records.
    void addSearchedMove(Pos move, Pos currentBestMove);

    /// Update move sorting heuristics when a best move is found.
    void updateBestmoveStats(Depth depth, Pos bestMove, Value bestValue);

    /// Update move sorting heuristics when return with a ttMove.
    void updateTTMoveStats(Depth depth, Pos ttMove, Value ttValue, Value beta);

    /// Updates move sorting heuristics for a quiet best move.
    void updateQuietStats(Pos move, int bonus);

private:
    const Board  &board;
    ABSearchData *searchData;
    SearchStack  *searchStack;
    int           attackCount = 0, quietCount = 0;
    Pos           attacksSearched[MAX_ATTACKS];
    Pos           quietsSearched[MAX_QUIETS];
};

}  // namespace Search::AB
