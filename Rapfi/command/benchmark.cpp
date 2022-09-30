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

#include "../core/hash.h"
#include "../core/iohelper.h"
#include "../core/pos.h"
#include "../core/types.h"
#include "../game/board.h"
#include "../search/hashtable.h"
#include "../search/searchthread.h"
#include "command.h"

#include <iostream>
#include <memory>
#include <vector>

struct BenchSet
{
    Rule             rule;
    int              boardSize;
    int              searchDepth;
    std::vector<Pos> position;
};

static const BenchSet benchSets[] = {
    {FREESTYLE, 20, 9, {Pos(7, 7), Pos(7, 6), Pos(7, 3), Pos(7, 10), Pos(6, 10)}},
    {FREESTYLE, 20, 11, {Pos(3, 2), Pos(4, 4), Pos(4, 5), Pos(5, 6)}},
    {FREESTYLE,
     15,
     12,
     {Pos(4, 1), Pos(5, 1), Pos(7, 2), Pos(6, 2), Pos(6, 5), Pos(7, 4), Pos(5, 8)}},
    {FREESTYLE, 15, 13, {Pos(10, 3), Pos(10, 4), Pos(9, 1)}},
    {FREESTYLE, 15, 14, {Pos(7, 9), Pos(10, 8), Pos(8, 11), Pos(9, 11), Pos(10, 12)}},
    {STANDARD, 15, 11, {Pos(3, 2), Pos(4, 4), Pos(4, 5), Pos(5, 6)}},
    {STANDARD, 15, 12, {Pos(7, 7), Pos(7, 6), Pos(7, 3), Pos(7, 10), Pos(6, 10)}},
    {RENJU, 15, 9, {Pos(6, 6), Pos(7, 7), Pos(8, 8)}},
    {RENJU, 15, 11, {Pos(3, 2), Pos(4, 4), Pos(4, 5), Pos(5, 6)}},
    {RENJU, 15, 12, {Pos(7, 7), Pos(7, 6), Pos(7, 3), Pos(7, 10), Pos(6, 10)}},
};

constexpr size_t TotalMoveTestNum = 1000000;
constexpr size_t TTSizeMB         = 16;

void Command::benchmark()
{
    std::unique_ptr<Board> board;
    size_t                 savedThreadNum        = Search::Threads.size();
    bool                   savedAspirationWindow = Config::AspirationWindow;
    bool    savedNumIterationAfterSingularRoot   = Config::NumIterationAfterSingularRoot;
    bool    savedNumIterationAfterMate           = Config::NumIterationAfterMate;
    MsgMode savedMessageMode                     = Config::MessageMode;
    Search::Threads.setNumThreads(1);

    // Benchmark for Board::move() and Board::undo()
    MESSAGEL("==========Move Bench==========");
    Time             startTime = now();
    size_t           moveCnt   = 0;
    constexpr size_t TestNum   = TotalMoveTestNum / arraySize(benchSets);
    for (const auto &benchSet : benchSets) {
        board = std::make_unique<Board>(benchSet.boardSize);
        board->newGame(benchSet.rule);

        for (size_t test = 0; test < TestNum; test++) {
            for (Pos p : benchSet.position)
                board->move(benchSet.rule, p);

            for (size_t i = 0; i < benchSet.position.size(); i++)
                board->undo(benchSet.rule);
        }
        moveCnt += TestNum * benchSet.position.size();
    }
    Time endTime  = now();
    Time duration = endTime - startTime;
    MESSAGEL("Total Time (ms): " << duration);
    MESSAGEL("Moves/s: " << moveCnt * 1000 / std::max<size_t>(duration, 1));

    MESSAGEL("=========Search Bench=========");
    Config::MessageMode                   = MsgMode::NONE;
    Config::AspirationWindow              = true;
    Config::NumIterationAfterSingularRoot = 0;
    Config::NumIterationAfterMate         = 0;
    Search::TT.resize(TTSizeMB * 1024);
    Search::SearchOptions options;
    options.infoMode            = Search::SearchOptions::INFO_NONE;
    options.disableOpeningQuery = true;
    duration                    = (Time)0;
    size_t searchNodes          = 0;

    Hash::XXHasher hasher(TTSizeMB);

    for (const auto &benchSet : benchSets) {
        board = std::make_unique<Board>(benchSet.boardSize);
        board->newGame(benchSet.rule);
        for (Pos p : benchSet.position)
            board->move(benchSet.rule, p);

        options.rule     = {benchSet.rule, GameRule::FREEOPEN};
        options.maxDepth = benchSet.searchDepth;
        Search::Threads.clear(true);
        startTime = now();
        Search::Threads.startThinking(*board, options, true);
        Search::Threads.waitForIdle();
        endTime = now();
        duration += endTime - startTime;

        size_t nodes = Search::Threads.nodesSearched();
        searchNodes += nodes;

        // Hash from nodes searched
        hasher << nodes;

        // Hash from entire TT
        hasher((void *)Search::TT.firstEntry(0), TTSizeMB * 1024 * 1024);
    }

    Config::MessageMode                   = savedMessageMode;
    Config::AspirationWindow              = savedAspirationWindow;
    Config::NumIterationAfterSingularRoot = savedNumIterationAfterSingularRoot;
    Config::NumIterationAfterMate         = savedNumIterationAfterMate;
    Search::Threads.setNumThreads(savedThreadNum);

    MESSAGEL("Total Time (ms): " << duration);
    MESSAGEL("Nodes: " << searchNodes);
    MESSAGEL("Nodes/s: " << searchNodes * 1000 / std::max<size_t>(duration, 1));
    MESSAGEL("Hash: " << std::hex << hasher << std::dec);
}
