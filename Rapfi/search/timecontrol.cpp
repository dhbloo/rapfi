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

#include "timecontrol.h"

#include "../config.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>

namespace {

/// Move importance regarding to game ply.
/// The curve is made from pure experience, better to be tuned with statistical data.
float moveImportance(int ply)
{
    constexpr float XScale = 8;
    constexpr float XShift = 50;
    constexpr float Skew   = 0.08;
    constexpr float OCoeff = 0.1;
    constexpr float OScale = 0.04;

    float basic        = std::pow(1 + std::exp((ply - XShift) / XScale), -Skew);
    float openingBonus = OCoeff * std::exp(-OScale * ply);

    return std::max(basic + openingBonus, 0.01f);
}

/// Divisor for max turn time. We generally risk less time on higher depth as search
/// time grows exponentially with the serach depth.
float timeDivisor(int depth)
{
    return Config::TimeDivisorBias
           + Config::TimeDivisorScale * std::pow(float(depth), Config::TimeDivisorDepthPow);
}

}  // namespace

namespace Search {

void TimeControl::init(Time turnTime, Time matchTime, Time matchTimeLeft, MoveParams params)
{
    startTime = now();
    if (matchTime == 0)  // unlimited match time
        matchTimeLeft = std::numeric_limits<Time>::max();

    float movesToGo = std::max(params.movesLeft, 1);
    maximumTime     = Time(matchTimeLeft / std::min(Config::MatchSpaceMin, movesToGo));
    maximumTime     = std::max(std::min(turnTime, maximumTime) - Config::TurnTimeReserved, (Time)0);
    ampleMatchTime  = turnTime * std::min(params.movesLeft, Config::MoveHorizon) < matchTimeLeft;
    optimumTime     = Time(maximumTime * Config::AdvancedStopRatio);

    if (!ampleMatchTime) {
        Time match  = Time(matchTimeLeft / std::min(Config::MatchSpace, movesToGo));
        Time turn   = Time(maximumTime / Config::AverageBranchFactor * moveImportance(params.ply));
        optimumTime = std::min(optimumTime, std::min(turn, match));
    }

    assert(optimum() <= maximum());
}

bool TimeControl::checkStop(IterParams params, float &timeReduction) const
{
    // If given ample match time, just see if we have used all optimum time
    if (ampleMatchTime)
        return elapsed() >= optimum();

    // Calculate a optimum turn time scale factor based on bestmove changes,
    // bestmove stability and eval oscillation
    float valueDropped  = params.prevBestValue - params.bestValue;
    float fallingFactor = Config::FallingFactorScale * valueDropped + Config::FallingFactorBias;
    fallingFactor       = std::clamp(fallingFactor, 0.5f, 1.5f);

    // If the bestMove is stable over several iterations, reduce time accordingly
    int bestMoveStablePly = params.depth - params.lastBestMoveChangeDepth;
    timeReduction         = 1.0 + Config::BestmoveStableReductionScale * bestMoveStablePly;
    float reduction =
        std::pow(params.prevTimeReduction, Config::BestmoveStablePrevReductionPow) / timeReduction;

    // Use part of the gained time from a previous stable move for the current move
    float bestMoveInstability = 1 + 2 * params.averageBestMoveChanges;

    // Calculate tweaked max turn time and optimum turn time
    Time maxTurn  = Time(maximum() / timeDivisor(params.depth));
    Time optiTurn = Time(optimum() * bestMoveInstability * reduction * fallingFactor);
    Time turnTime = std::min(optiTurn, maxTurn);

    return elapsed() >= turnTime;
}

bool TimeControl::checkStop(PlayoutParams params) const
{
    return elapsed() >= optimum();
}

}  // namespace Search
