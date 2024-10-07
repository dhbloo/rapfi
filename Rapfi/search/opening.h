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
#include "../game/board.h"

namespace Opening {

/// Try to probe a opening according to the certain rule.
/// @param[inout] board Board state to probe a opening move. New move(s) will be put on board.
/// @param[in] rule The rule including game rule and opening rule.
/// @param[out] action The result action of probed opening.
/// @param[out] move The generated move.
/// @return Whether there is a suitable opening. If not, the engine needs
///     to search the position to return a move.
bool probeOpening(Board &board, GameRule rule, ActionType &action, Pos &move);

/// Decide final action according to game rule and search result value.
ActionType decideAction(const Board &board, GameRule rule, Value bestValue);

/// Check if there are any moves near border and expands board candidate
/// area for those moves.
void expandCandidate(Board &board);

/// Expand half of the board as candidates. Used to find balance move
/// when the board is empty.
void expandCandidateHalfBoard(Board &board);

/// Check if the board is symmetry under the given transform.
bool isBoardSymmetry(const Board &board, TransformType symTrans);

/// Remove redundant symmetry moves from move list.
void filterSymmetryMoves(const Board &board, std::vector<Pos> &moveList);

/// OpeningGenConfig struct contains information on how to generate a opening.
struct OpeningGenConfig
{
    // How many moves one opening can have (0 < minMoves < maxMoves)
    int minMoves = 2, maxMoves = 10;

    // Randomly picked moves are chosen inside a local square with size
    // of [localSizeMin, localSizeMax]
    int localSizeMin = 4, localSizeMax = 6;

    // Find a balanced opening using search time limit of balanceNodes.
    // If it is 0, then the generated openings will not be balanced.
    uint64_t balance1Nodes = 1000000;

    // Use how many nodes to fast check if this position is balanceable.
    uint64_t balance1FastCheckNodes = 100000;

    // When BALANCE1 is not able to find a balanced move, use how many
    // extra nodes to find a balanced move pair in BALANCE2.
    uint64_t balance2Nodes = 2500000;

    // Eval in [-balanceWindow, balanceWindow] is considered as balanced
    // When we can not achieve a eval inside balance window using BALANCE1,
    // BALANCE2 is tried instead;
    Value balanceWindow = Value(50);

    // Consider this position as unbalanceable if its initial search value
    // falls outside this window.
    Value balance1FastCheckWindow = Value(120);
};

/// OpeningGenerator class generates various (balanced) random opening using
/// opening generation config. The generation process contains two phases:
/// the first is to randomly put moves on a random local rectangle area on
/// board, the second is to try put one/two balance move(s) which makes it
/// giving the most balanced eval within the limited search time.
class OpeningGenerator
{
public:
    OpeningGenerator(int boardSize, Rule rule, OpeningGenConfig config = {}, PRNG prng = {});

    const Board &getBoard() const { return board; }
    std::string  positionString() const { return board.positionString(); }
    bool         next();

private:
    Board            board;
    Rule             rule;
    OpeningGenConfig config;
    PRNG             prng;
    std::vector<int> numMoveChoices;

    void putRandomMoves(int numMoves, CandArea area);
    bool putBalance1Move();
    bool putBalance2Move();
};

}  // namespace Opening
