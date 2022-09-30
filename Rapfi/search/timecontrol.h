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

#include "../core/types.h"
#include "../core/utils.h"

namespace Search {

/// TimeControl class computes the optimal and maximum turn time
/// depending on maxTurnTime, matchTimeLeft and some other parameters.
class TimeControl
{
public:
    /// MoveParams struct contains all info needed per move
    /// to calculate optimum and maximum turn time.
    struct MoveParams
    {
        int ply;
        int movesLeft;
    };

    /// IterParams struct contains all info needed per depth in
    /// iterative deepening search to adjust optimum turn time.
    struct IterParams
    {
        int   depth;
        int   lastBestMoveChangeDepth;
        Value bestValue;
        Value prevBestValue;
        float prevTimeReduction;
        float averageBestMoveChanges;
    };

    void init(Time maxTurnTime, Time matchTimeLeft, MoveParams params);
    bool checkStop(IterParams params, float &timeReduction) const;

    Time optimum() const { return optimumTime; }
    Time maximum() const { return maximumTime; }
    Time elapsed() const { return now() - startTime; }

private:
    Time startTime;
    Time optimumTime;
    Time maximumTime;

    /// Whether the match time is much longer than the turn time that we will never
    /// spend all the match time even if we spend all turn time at each turn.
    bool ample;
};

}  // namespace Search
