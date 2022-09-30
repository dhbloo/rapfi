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

#include "../core/pos.h"
#include "../core/types.h"
#include "../core/utils.h"
#include "searchthread.h"

#include <random>

namespace Search {

/// SkillMovePicker class implements strength handicap by introducing randomness
/// to select best move according to their eval and strength level.
class SkillMovePicker
{
public:
    SkillMovePicker(int level) : level(level)
    {
        constexpr int    BaseDepth   = 4;
        constexpr int    FullDepth   = 16;
        constexpr double Alpha       = 0.5;
        constexpr double MaxWeakness = 0.9;
        constexpr double MinWeakness = 0.1;

        double k    = (FullDepth - BaseDepth) * (1. / (Alpha - 1));
        double t    = level * (1. / 100.);
        targetDepth = BaseDepth + int(k * (std::pow(Alpha, t) - 1));
        weakness    = MinWeakness * t + MaxWeakness * (1.0 - t);
    };
    bool     enabled() const { return level < 100; }
    int      pickDepth() const { return targetDepth; }
    uint32_t minMultiPv() const { return 2 + (100 - level) / 25; }
    Pos      pick(ThreadPool &threads, size_t numMultiPV = 1);

private:
    static constexpr Value MaxRandomRange = Value(120);

    const int level;
    int       targetDepth;
    double    weakness;
};

/// Select a move among multiple root moves using a statistical rule
/// dependent on 'level' (code from stockfish).
inline Pos SkillMovePicker::pick(ThreadPool &threads, size_t numMultiPV)
{
    const auto &rootMoves = threads.main()->rootMoves;
    static PRNG prng;  // init seed from time to ensure randomness

    // Init highest value and value range
    Value topValue = rootMoves[0].value;
    int   range    = std::min(topValue - rootMoves[numMultiPV - 1].value, MaxRandomRange);

    // Pick best move by comparing its value plus a random bonus
    std::uniform_real_distribution<> dis(0.0, weakness);
    Value                            maxValue = -VALUE_INFINITE;
    Pos                              best     = rootMoves[0].pv[0];
    for (size_t i = 0; i < numMultiPV; i++) {
        // Add a random bonus depending on weakness and value range
        int   bonus     = int(weakness * int(topValue - rootMoves[i].value) + dis(prng) * range);
        Value moveValue = rootMoves[i].value + bonus;

        if (moveValue > maxValue) {
            maxValue = moveValue;
            best     = rootMoves[i].pv[0];
        }
    }

    return best;
}

}  // namespace Search
