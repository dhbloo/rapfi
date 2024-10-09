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
#include "argutils.h"
#include "command.h"

#include <iostream>
#include <memory>
#include <vector>

constexpr size_t         TotalMoveTestNum = 2000000;
constexpr size_t         TTSizeMB         = 16;
constexpr CandidateRange CandRange        = CandidateRange::SQUARE3_LINE4;

struct BenchEntry
{
    Rule        rule;
    int         boardSize;
    int         searchDepth;
    std::string positionString;
};

static const std::vector<BenchEntry> benchSet = {
    // Freestyle positions
    {FREESTYLE, 15, 17, "h8g7f6g8g9"},
    {FREESTYLE, 15, 19, "h2h5h4f6"},
    {FREESTYLE, 20, 17, "e2f2h3g3g6h5f9"},
    {FREESTYLE, 20, 19, "f5e3b7h5g3g4i6e4h4d4f2i5f4f6e5d6d7e7f8"},

    // Standard positions
    {STANDARD, 15, 17, "h8h7j6i7f7"},
    {STANDARD, 15, 18, "f6i9h8"},
    {STANDARD, 15, 19, "b3k10g2f6"},

    // Renju positions
    {RENJU, 15, 17, "h8i9j10i8i7g9"},
    {RENJU, 15, 18, "h8h9j9j8f8i7g7i9h6e9f6k7l6i10i6f9g9g11h10i11i8"},
    {RENJU, 15, 19, "h8h9h6i10i6i9g9g8j11i7"},
};

struct EngineState
{
    size_t  threadNum;
    size_t  memoryLimitKB;
    bool    aspirationWindow;
    int     numIterationAfterSingularRoot;
    int     numIterationAfterMate;
    MsgMode messageMode;
};

EngineState saveEngineStateForBenckmark()
{
    EngineState state;

    state.threadNum                     = Search::Threads.size();
    state.memoryLimitKB                 = Search::Threads.searcher()->getMemoryLimit();
    state.aspirationWindow              = Config::AspirationWindow;
    state.numIterationAfterSingularRoot = Config::NumIterationAfterSingularRoot;
    state.numIterationAfterMate         = Config::NumIterationAfterMate;
    state.messageMode                   = Config::MessageMode;

    return state;
}

void recoverEngineState(EngineState state)
{
    Search::Threads.setNumThreads(state.threadNum);
    Search::Threads.searcher()->setMemoryLimit(state.memoryLimitKB);
    Config::MessageMode                   = state.messageMode;
    Config::AspirationWindow              = state.aspirationWindow;
    Config::NumIterationAfterSingularRoot = state.numIterationAfterSingularRoot;
    Config::NumIterationAfterMate         = state.numIterationAfterMate;
}

void Command::benchmark()
{
    std::unique_ptr<Board> board;
    EngineState            backupState = saveEngineStateForBenckmark();

    // Benchmark for Board::move() and Board::undo()
    MESSAGEL("==========Move Bench==========");
    Time   duration        = 0;
    size_t moveCount       = 0;
    size_t testNumPerEntry = TotalMoveTestNum / benchSet.size();
    for (const auto &benchEntry : benchSet) {
        board = std::make_unique<Board>(benchEntry.boardSize, CandRange);
        board->newGame(benchEntry.rule);
        std::vector<Pos> position =
            parsePositionString(benchEntry.positionString, board->size(), board->size());

        Time   startTime = now();
        size_t testNum   = testNumPerEntry / position.size();
        for (size_t test = 0; test < testNum; test++) {
            for (size_t i = 0; i < position.size(); i++)
                board->move(benchEntry.rule, position[i]);

            for (size_t i = 0; i < position.size(); i++)
                board->undo(benchEntry.rule);
        }
        Time endTime = now();

        duration += endTime - startTime;
        moveCount += testNum * position.size();
    }

    MESSAGEL("Total Time (ms): " << duration);
    MESSAGEL("Moves/s: " << moveCount * 1000 / std::max<size_t>(duration, 1));

    MESSAGEL("=========Search Bench=========");
    Config::MessageMode                   = MsgMode::NONE;
    Config::AspirationWindow              = true;
    Config::NumIterationAfterSingularRoot = 0;
    Config::NumIterationAfterMate         = 0;
    Search::Threads.setNumThreads(1);
    Search::Threads.searcher()->setMemoryLimit(TTSizeMB * 1024);
    Search::SearchOptions options;
    options.infoMode            = Search::SearchOptions::INFO_NONE;
    options.disableOpeningQuery = true;
    duration                    = 0;
    size_t searchNodes          = 0;

    Hash::XXHasher hasher(TTSizeMB);

    for (const auto &benchEntry : benchSet) {
        board = std::make_unique<Board>(benchEntry.boardSize, CandRange);
        board->newGame(benchEntry.rule);
        std::vector<Pos> position =
            parsePositionString(benchEntry.positionString, board->size(), board->size());

        for (Pos p : position)
            board->move(benchEntry.rule, p);

        options.rule     = {benchEntry.rule, GameRule::FREEOPEN};
        options.maxDepth = benchEntry.searchDepth;
        Search::Threads.clear(true);

        Time startTime = now();
        Search::Threads.startThinking(*board, options, true);
        Search::Threads.waitForIdle();
        Time endTime = now();

        duration += endTime - startTime;

        size_t nodes = Search::Threads.nodesSearched();
        searchNodes += nodes;

        // Hash from nodes searched
        hasher << nodes;
        // Hash from the last output eval
        hasher << Search::Threads.main()->rootMoves[0].value;
    }

    uint32_t hash32 = (uint64_t(hasher) >> 32) ^ uint64_t(hasher);
    MESSAGEL("Total Time (ms): " << duration);
    MESSAGEL("Nodes: " << searchNodes);
    MESSAGEL("Nodes/s: " << searchNodes * 1000 / std::max<size_t>(duration, 1));
    MESSAGEL("Hash: " << std::hex << hash32 << std::dec);

    recoverEngineState(backupState);
}
