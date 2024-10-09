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

#include "../core/pos.h"
#include "../core/types.h"
#include "../core/utils.h"

#include <cassert>
#include <cstdlib>
#include <vector>

class Board;

namespace Search {

/// Make a balanced value from original value and balance bias.
/// @param value The original value.
/// @param bias Value bias for balanced value.
/// @return Comparable balanced value.
inline Value balancedValue(Value value, int bias)
{
    return Value(-std::abs((int)value - bias) + bias);
}

/// Balance2Move struct stores a move pair in balance2 move mode.
struct Balance2Move
{
    Pos move1, move2;

    bool operator==(const Balance2Move &m) const { return move1 == m.move1 && move2 == m.move2; }

    /// A hasher is implemented for using Balance2Move as a key in std::unordered_map.
    struct Hash
    {
        size_t operator()(const Balance2Move &m) const noexcept
        {
            return std::hash<size_t> {}((size_t)m.move1 << 16 | (size_t)m.move2);
        }
    };
};

/// RootMove struct is used for moves at the root of the tree. For each root
/// move we store a value and a PV (really a refutation in the case of moves
/// which fail low). Value is set at -VALUE_INFINITE for all non-pv moves
/// before each iteration so that pv are.
struct RootMove
{
    RootMove(Pos pos) : pv(1, pos), previousPv(pv) {}
    RootMove(Balance2Move m) : pv {m.move1, m.move2}, previousPv(pv) {}
    bool operator==(const Pos &m) const { return pv[0] == m; }
    bool operator==(const Balance2Move &m) const
    {
        assert(pv.size() >= 2);
        return pv[0] == m.move1 && pv[1] == m.move2;
    }

    Value    value          = VALUE_NONE;
    Value    previousValue  = VALUE_NONE;
    int      selDepth       = 0;
    float    winRate        = std::numeric_limits<float>::quiet_NaN();
    float    drawRate       = std::numeric_limits<float>::quiet_NaN();
    float    policyPrior    = std::numeric_limits<float>::quiet_NaN();
    float    utilityStdev   = std::numeric_limits<float>::quiet_NaN();
    float    lcbValue       = std::numeric_limits<float>::quiet_NaN();
    float    selectionValue = std::numeric_limits<float>::quiet_NaN();
    uint64_t numNodes       = 0;

    std::vector<Pos> pv, previousPv;
};

using RootMoves = std::vector<RootMove>;

/// RootMoveValueComparator compares two root moves according to their value, and
/// sorts them in descending order.
struct RootMoveValueComparator
{
    bool operator()(const RootMove &a, const RootMove &b) const;
};

/// BalanceMoveValueComparator compares two root moves according to their
/// distance to the balanced eval in balance move mode.
struct BalanceMoveValueComparator
{
    int bias = 0;

    /// Compare two root moves in balance mode.
    bool operator()(const RootMove &a, const RootMove &b) const;
};

/// SearchOptions stores information sent by GUI, including game rule,
/// maximum depth/time/node, analysis mode, searched pv count, etc.
struct SearchOptions
{
    // Game rules
    GameRule rule = {Rule::FREESTYLE, GameRule::FREEOPEN};
    // Whether current move is swapable.
    // If true, when current position matches opening rule, swap is considered
    bool swapable = false;
    /// Turn off opening book probing.
    bool disableOpeningQuery = false;
    /// Enable background pondering after each search.
    bool pondering = false;
    /// Enable time control infomation to terminate search.
    bool timeLimit = false;

    /// Output realtime thinking process.
    enum InfoMode : uint8_t {
        INFO_NONE                = 0b00,
        INFO_REALTIME            = 0b01,
        INFO_DETAIL              = 0b10,
        INFO_REALTIME_AND_DETAIL = 0b11,
    } infoMode = INFO_NONE;

    // Time control
    Time     turnTime   = 0;
    Time     matchTime  = 0;
    Time     timeLeft   = 0;
    uint64_t maxNodes   = 0;  // (0 means no limits)
    int      maxDepth   = 99;
    int      startDepth = 2;

    /// MultiPV mode, normally set to 1
    uint16_t multiPV = 1;
    /// Playing strength control (0~100, 100 for maximum strength level)
    uint16_t strengthLevel = 100;
    /// Balance mode (default is BALANCE_NONE)
    enum BalanceMode {
        BALANCE_NONE,
        BALANCE_ONE,
        BALANCE_TWO,
    } balanceMode = BALANCE_NONE;
    /// Value bias for balance move (default is 0)
    int balanceBias = 0;
    /// Max game ply for the search tree to grow
    int maxMoves = INT32_MAX;
    /// The result returned after reaching the max game ply
    enum DrawResult {
        RES_DRAW,
        RES_BLACK_WIN,
        RES_WHITE_WIN,
    } drawResult = RES_DRAW;

    /// Blocked moves, which are filtered out before searching
    std::vector<Pos> blockMoves;

    /// Checks if we are in analysis mode.
    bool isAnalysisMode() const { return !timeLimit && !maxNodes; }
    /// Set time control config according to the rule:
    ///     MatchTime (ms) |              TurnTime (ms)
    ///                    | less than 0 | equal to 0 | more than 0
    ///        less than 0 |  Infinite   |  Infinite  | Turn only
    ///         equal to 0 |  Infinite   |  Infinite  | Turn only
    ///        more than 0 |  Match only | Match only | Match+Turn
    void setTimeControl(int64_t turnTime, int64_t matchTime);
};

/// Get the therotical game value after reaching the max game ply.
Value getDrawValue(const Board &board, const SearchOptions &options, int ply);

}  // namespace Search
