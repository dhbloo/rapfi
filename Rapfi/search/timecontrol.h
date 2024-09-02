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

    /// PlayoutParams struct contains all info needed per playout
    /// in MCTS search to adjust optimum turn time.
    struct PlayoutParams
    {};

    /// Compute the optimal and maximum turn time at the beginning of a search.
    /// @param turnTime Max turn time from search options.
    /// @param matchTime Max match time from search options.
    /// @param matchTimeLeft Current match time left from search options.
    /// @param params Parameters from current move to search.
    void init(Time turnTime, Time matchTime, Time matchTimeLeft, MoveParams params);

    /// Check if we need to stop iterating deepening at this depth.
    /// @param[in] params The time parameters from last iteration.
    /// @param[out] timeReduction Record how much time is saved in the last move.
    /// @return True if we should stop the search.
    bool checkStop(IterParams params, float &timeReduction) const;

    /// Check if we need to stop iterating deepening at this depth.
    /// @param[in] params The time parameters from last playout.
    /// @return True if we should stop the search.
    bool checkStop(PlayoutParams params) const;

    Time optimum() const { return optimumTime; }
    Time maximum() const { return maximumTime; }
    Time elapsed() const { return now() - startTime; }

private:
    Time startTime;
    Time optimumTime;
    Time maximumTime;

    /// Whether the match time is much longer than the turn time that we will never
    /// spend all the match time even if we spend all turn time at each turn.
    bool ampleMatchTime;
};

}  // namespace Search
