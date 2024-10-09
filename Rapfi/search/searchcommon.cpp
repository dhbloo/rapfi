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

#include "searchcommon.h"

#include "../config.h"
#include "../game/board.h"

namespace Search {

bool RootMoveValueComparator::operator()(const RootMove &a, const RootMove &b) const
{
    return a.value != b.value ? a.value > b.value : a.previousValue > b.previousValue;
}

bool BalanceMoveValueComparator::operator()(const RootMove &a, const RootMove &b) const
{
    return a.value != b.value
               ? balancedValue(a.value, bias) > balancedValue(b.value, bias)
               : balancedValue(a.previousValue, bias) > balancedValue(b.previousValue, bias);
}

void SearchOptions::setTimeControl(int64_t turnTime, int64_t matchTime)
{
    if (turnTime <= 0 && matchTime <= 0) {  // Infinite time
        this->turnTime  = 0;
        this->matchTime = 0;
        this->timeLimit = false;
    }
    else if (turnTime > 0 && matchTime <= 0) {  // Turn time only
        this->turnTime  = turnTime;
        this->matchTime = 0;
        this->timeLimit = true;
    }
    else if (turnTime <= 0) {  // Match time only
        this->turnTime  = matchTime;
        this->matchTime = matchTime;
        this->timeLimit = true;
    }
    else {  // Match time + turn time
        this->turnTime  = turnTime;
        this->matchTime = matchTime;
        this->timeLimit = true;
    }

    this->timeLeft = this->matchTime;
}

/// Get the return value after reaching the max game ply.
Value getDrawValue(const Board &board, const SearchOptions &options, int ply)
{
    int pliesUntilMaxPly = std::max(options.maxMoves - board.nonPassMoveCount(), 0);
    int matePly          = ply + pliesUntilMaxPly;

    switch (options.drawResult) {
    default: return VALUE_DRAW;
    case SearchOptions::RES_BLACK_WIN:
        return board.sideToMove() == BLACK ? mate_in(matePly) : mated_in(matePly);
    case SearchOptions::RES_WHITE_WIN:
        return board.sideToMove() == WHITE ? mate_in(matePly) : mated_in(matePly);
    }
}

}  // namespace Search
