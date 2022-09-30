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

#include "../game/board.h"
#include "../core/pos.h"
#include "../core/types.h"

namespace Opening {

bool probeOpening(Board &board, GameRule rule, ActionType &action, Pos &move);

ActionType decideAction(const Board &board, GameRule rule, Value bestValue);

void expandCandidate(Board &board);

void expandCandidateHalfBoard(Board &board);

void filterSymmetryMoves(const Board &board, std::vector<Pos> &moveList);

/// OpeningGenConfig struct contains information on how to generate a opening.
struct OpeningGenConfig
{
    // How many moves one opening can have (0 < minMoves < maxMoves)
    int minMoves = 1, maxMoves = 10;

    // Randomly picked moves are chosen inside a local square with size
    // of [localSizeMin, localSizeMax]
    int localSizeMin = 4, localSizeMax = 8;

    // Find a balanced opening using search time limit of balanceNodes.
    // If it is 0, then the generated openings will not be balanced.
    uint64_t balance1Nodes = 500000;
    // When BALANCE1 is not able to find a balanced move, use how many
    // extra nodes to find a balanced move pair in BALANCE2.
    uint64_t balance2Nodes = 1000000;

    // Eval in [-balanceWindow, balanceWindow] is considered as balanced
    // When we can not achieve a eval inside balance window using BALANCE1,
    // BALANCE2 is tried instead;
    Value balanceWindow = Value(50);
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
