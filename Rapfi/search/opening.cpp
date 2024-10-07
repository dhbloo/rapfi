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

#include "opening.h"

#include "../core/iohelper.h"
#include "../game/board.h"
#include "search.h"
#include "searchthread.h"

#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

/// Check if a move is near border by some distance.
/// @param dist Distance from border that is considered as near.
bool isNearBorder(const Board &board, Pos pos, int dist)
{
    int x = pos.x(), y = pos.y();
    int min = dist, max = board.size() - 1 - dist;
    return x <= min || y <= min || x >= max || y >= max;
}

/// Generate opening move for (almost) empty board state.
/// @param[in] board Board state to probe a opening move.
/// @param[in] rule The rule including game rule and opening rule.
/// @return The generated moves, empty if no opening is available.
std::vector<Pos> generateOpening(const Board &board, GameRule rule)
{
    PRNG prng;

    switch (rule.opRule) {
    case GameRule::FREEOPEN:
        if (board.ply() == 0)
            return {board.centerPos()};
        break;
    case GameRule::SWAP1:
        if (board.ply() == 0 && board.size() >= 13) {
            int c = board.size() / 2;

            const std::vector<std::pair<int, int>> balanceCoordIncrements = {
                std::make_pair(c, 1),
                std::make_pair(c - 1, 1),
                std::make_pair(c - 2, 1),
            };
            std::uniform_int_distribution<> choice(0, balanceCoordIncrements.size() - 1);
            std::uniform_int_distribution<> binary(0, 1);
            auto                            coord = balanceCoordIncrements[choice(prng)];

            if (binary(prng))  // swap x and y coordinate
                coord = std::make_pair(coord.second, coord.first);

            int x = binary(prng) ? coord.first : board.size() - 1 - coord.first;
            int y = binary(prng) ? coord.second : board.size() - 1 - coord.second;

            return {Pos {x, y}};
        }
        break;
    case GameRule::SWAP2:
        if (board.ply() == 0 && board.size() == 15) {
            const std::vector<std::vector<Pos>> openings = {
                {Pos(6, 7), Pos(6, 4), Pos(4, 2)},
                {Pos(3, 3), Pos(5, 5), Pos(6, 6)},
                {Pos(3, 2), Pos(5, 4), Pos(4, 5)},
                {Pos(5, 2), Pos(1, 5), Pos(1, 6)},
                {Pos(8, 5), Pos(5, 8), Pos(6, 7)},
                {Pos(5, 5), Pos(8, 8), Pos(7, 7)},
                {Pos(13, 12), Pos(13, 9), Pos(10, 12)},
                {Pos(11, 7), Pos(10, 6), Pos(13, 5)},
                {Pos(3, 7), Pos(1, 8), Pos(0, 4)},
            };

            std::uniform_int_distribution<> choice(0, openings.size() - 1);
            return openings[choice(prng)];
        }
        break;
    }

    return {};
}

/// A simple implementation of disjoint set (union-find).
class IndexDisjointSet
{
public:
    explicit IndexDisjointSet(size_t size) noexcept : root(size), size(size, 1), numSets(size)
    {
        std::iota(root.begin(), root.end(), 0);
    }

    size_t find(size_t x) noexcept
    {
        assert(isValid(x));

        if (root[x] != x)
            return root[x] = find(root[x]);
        else
            return x;
    }

    template <typename... Args>
    void merge(size_t x, size_t y, Args... tail) noexcept
    {
        assert(isValid(x) && isValid(y));
        unite(x, y);
        if constexpr (sizeof...(tail) > 0) {
            merge(y, tail...);
        }
    }

    template <typename... Args>
    bool areConnected(size_t x, size_t y, Args... tail) noexcept
    {
        assert(isValid(x) && isValid(y));

        x              = find(x);
        y              = find(y);
        bool connected = x == y;
        if constexpr (sizeof...(tail) == 0)
            return connected;
        else
            return connected && areConnected(y, tail...);
    }

    size_t sizeOfSetContaining(size_t x) noexcept { return size[find(x)]; }
    size_t setCount() const noexcept { return numSets; }
    bool   isValid(size_t x) const noexcept { return x < root.size(); }

private:
    std::vector<size_t> root;
    std::vector<size_t> size;
    size_t              numSets;

    void unite(size_t x, size_t y) noexcept
    {
        x = find(x);
        y = find(y);

        if (x == y)
            return;

        numSets--;

        if (size[x] < size[y])
            std::swap(x, y);

        root[y] = x;
        size[x] += size[y];
    }
};

}  // namespace

namespace Opening {

bool probeOpening(Board &board, GameRule rule, ActionType &action, Pos &move)
{
    std::vector<Pos> openingMoves = generateOpening(board, rule);
    if (openingMoves.empty())
        return false;

    if (openingMoves[0] == Pos(-1, -1))
        action = ActionType::Swap;
    else {
        action = ActionType::Move;
        move   = openingMoves[0];
        for (Pos move : openingMoves)
            board.move(rule, move);
    }

    return true;
}

ActionType decideAction(const Board &board, GameRule rule, Value bestValue)
{
    switch (rule.opRule) {
    case GameRule::SWAP1:
        if (board.ply() == 1 && bestValue < 0)
            return ActionType::Swap;
        break;
    case GameRule::SWAP2:
        if (board.ply() == 3) {
            if (bestValue < 0)
                return ActionType::Swap;
        }
        else if (board.ply() == 5 && bestValue < 0)
            return ActionType::Swap;
        break;
    default: break;
    }

    return ActionType::Move;
}

void expandCandidate(Board &board)
{
    for (int i = 0; i < board.ply(); i++) {
        Pos pos = board.getHistoryMove(i);

        if (isNearBorder(board, pos, 0))
            board.expandCandArea(pos, 4, 0);
        else if (isNearBorder(board, pos, 2))
            board.expandCandArea(pos, 3, 0);
        else
            break;
    }
}

void expandCandidateHalfBoard(Board &board)
{
    int c = board.size() / 4;
    board.expandCandArea(Pos {c, c}, c + 1, 0);
}

bool isBoardSymmetry(const Board &board, TransformType symTrans)
{
    FOR_EVERY_POSITION(&board, pos)
    {
        if (board.isEmpty(pos))
            continue;

        Pos transformedPos = applyTransform(pos, board.size(), symTrans);
        if (board.get(pos) != board.get(transformedPos))
            return false;
    }

    return true;
}

void filterSymmetryMoves(const Board &board, std::vector<Pos> &moveList)
{
    std::unordered_map<Pos, size_t> posToIndexMap;
    std::unordered_map<size_t, Pos> indexToPosMap;
    size_t                          index = 0;
    FOR_EVERY_EMPTY_POS(&board, pos)
    {
        posToIndexMap[pos]   = index;
        indexToPosMap[index] = pos;
        index++;
    }
    IndexDisjointSet ds(index);

    for (int i = 0; i < TRANS_NB; i++) {
        TransformType trans = (TransformType)i;

        if (!isBoardSymmetry(board, trans))
            continue;

        FOR_EVERY_EMPTY_POS(&board, pos)
        {
            Pos transformedPos = applyTransform(pos, board.size(), trans);
            ds.merge(posToIndexMap[pos], posToIndexMap[transformedPos]);
        }
    }

    // Remove all redundant symmetry moves (which is not root in ds)
    auto pred = [&](Pos move) -> bool {
        if (posToIndexMap.find(move) == posToIndexMap.end())
            return true;

        size_t idx = posToIndexMap[move];
        return ds.find(idx) != idx;
    };
    moveList.erase(std::remove_if(moveList.begin(), moveList.end(), pred), moveList.end());
}

/// Create a opening generator, which can be used to generate balanced
/// opening with the given board size, rule and config.
/// @param boardSize The size of board to generate openings.
/// @param rule The rule used to generate openings.
/// @param cfg The config object which specifies how to generate openings.
OpeningGenerator::OpeningGenerator(int boardSize, Rule rule, OpeningGenConfig cfg, PRNG prng)
    : board(boardSize)
    , rule(rule)
    , config(cfg)
    , prng(prng)
{
    assert(config.minMoves <= config.maxMoves);
    for (int nMoves = config.minMoves; nMoves <= config.maxMoves; nMoves++) {
        for (int i = 0; i < nMoves; i++)
            numMoveChoices.push_back(nMoves);
    }
}

/// Generate the next opening. The opening is saved in the internal board
/// state, which can be accessed via cloneBoard() or positionString().
/// @return Whether it generated a opening with eval in balance window
bool OpeningGenerator::next()
{
    board.newGame(rule);

    // Pick a random move count
    bool findBalancedMove = config.balance1Nodes || config.balance2Nodes;
    int  idx              = std::uniform_int_distribution<int>(0, numMoveChoices.size() - 1)(prng);
    int  numMoves         = numMoveChoices[idx];

    // Spare two moves for balanced position (if available)
    int numRandomMoves = std::max(numMoves - findBalancedMove, 0);
    if (numRandomMoves > 0) {
        // Put random moves on board
        int areaSize =
            std::uniform_int_distribution<>(config.localSizeMin, config.localSizeMax)(prng);
        int x0 = std::uniform_int_distribution<>(0, board.size() - 1 - areaSize)(prng);
        int y0 = std::uniform_int_distribution<>(0, board.size() - 1 - areaSize)(prng);

        CandArea area(x0, y0, x0 + areaSize, y0 + areaSize);
        putRandomMoves(numRandomMoves, area);
    }

    if (!findBalancedMove)
        return false;

    // For empty board, randomly make a part of the whole board as move condidate
    if (board.ply() == 0)
        board.expandCandArea(
            board.centerPos(),
            std::uniform_int_distribution<>(board.size() / 4, board.size() / 2 + 1)(prng),
            0);

    // First try to put balance1 move, if that fails, try balance2
    if (config.balance1Nodes) {
        if (Config::MessageMode != MsgMode::NONE)
            MESSAGEL("Searching balanced1 for opening " << board.positionString());
        if (putBalance1Move()) {
            board.move(rule, Search::Threads.main()->rootMoves[0].pv[0]);
            return true;
        }
    }

    if (config.balance2Nodes && board.ply() > 1) {
        board.undo(rule);  // undo one random move

        if (Config::MessageMode != MsgMode::NONE)
            MESSAGEL("Searching balanced2 for opening " << board.positionString());
        if (putBalance2Move()) {
            board.move(rule, Search::Threads.main()->rootMoves[0].pv[0]);
            board.move(rule, Search::Threads.main()->rootMoves[0].pv[1]);
            return true;
        }
    }

    return false;
}

/// Put random moves in the given area on board.
/// @param numMoves The number of random moves to put.
/// @param area The candidate area which random moves can be put in.
/// @note If the size of candidate area has less move than the given number
///     of random moves, the actual moves put will be the size of that area.
void OpeningGenerator::putRandomMoves(int numMoves, CandArea area)
{
    // Collect all candidate moves in the area and shuffle them
    std::vector<Pos> randomMoves;
    FOR_EVERY_CANDAREA_POS(&board, pos, area)
    {
        randomMoves.push_back(pos);
    }
    std::shuffle(randomMoves.begin(), randomMoves.end(), prng);

    int count = 0;
    for (Pos pos : randomMoves) {
        board.move(rule, pos);

        // Make sure we do not accidentally put a winning move
        if (board.p4Count(board.sideToMove(), A_FIVE) || board.p4Count(board.sideToMove(), B_FLEX4)
            || board.p4Count(board.sideToMove(), C_BLOCK4_FLEX3)) {
            board.undo(rule);
        }
        else {
            count++;
            if (count >= numMoves)
                break;
        }
    }
}

/// Put one move to make a balanced opening.
/// @return True if successfully found a balanced move.
bool OpeningGenerator::putBalance1Move()
{
    using Search::Threads;

    // Setup search options
    Search::SearchOptions options;
    options.rule.rule           = rule;
    options.disableOpeningQuery = true;
    options.balanceMode         = Search::SearchOptions::BALANCE_ONE;

    Threads.clear(false);
    assert(!Threads.empty());

    // First do fast check to see if this position is balanceable
    options.maxNodes = config.balance1FastCheckNodes;
    if (options.maxNodes > 0) {
        Threads.startThinking(board, options);
        Threads.waitForIdle();

        // Consider this position as unbalanceable if its initial value is too far
        if (std::abs(Threads.main()->rootMoves[0].value) > config.balance1FastCheckWindow)
            return false;
    }

    // Start the actual BALANCE1 search
    options.maxNodes = std::max(config.balance1Nodes, config.balance1FastCheckNodes)
                       - config.balance1FastCheckNodes;
    if (options.maxNodes > 0) {
        Threads.startThinking(board, options);
        Threads.waitForIdle();
    }

    // Check if BALANCE1 has find a move that is balanced enough
    assert(!Threads.main()->rootMoves.empty());
    return std::abs(Threads.main()->rootMoves[0].value) <= config.balanceWindow;
}

/// Put two moves to make a balanced opening.
/// @return True if successfully found a balanced move pair.
bool OpeningGenerator::putBalance2Move()
{
    using Search::Threads;

    // Setup search options
    Search::SearchOptions options;
    options.rule.rule           = rule;
    options.disableOpeningQuery = true;
    options.balanceMode         = Search::SearchOptions::BALANCE_TWO;
    options.maxNodes            = config.balance2Nodes;

    Threads.clear(false);
    assert(!Threads.empty());

    // Try BALANCE2 search
    Threads.startThinking(board, options);
    Threads.waitForIdle();

    assert(!Threads.main()->rootMoves.empty());
    return std::abs(Threads.main()->rootMoves[0].value) <= config.balanceWindow;
}

}  // namespace Opening
