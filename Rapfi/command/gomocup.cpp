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

#include "../config.h"
#include "../core/iohelper.h"
#include "../core/utils.h"
#include "../database/dbconfig.h"
#include "../eval/eval.h"
#include "../eval/evalconfig.h"
#include "../eval/evaluator.h"
#include "../game/board.h"
#include "../search/hashtable.h"
#include "../search/movepick.h"
#include "../search/opening.h"
#include "../search/searcher.h"
#include "../search/searchthread.h"
#include "../tuning/tunemap.h"
#include "command.h"
#include "dbcommand.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>

#ifdef MULTI_THREADING
    #include <mutex>
    #include <thread>

static std::mutex protocolMutex;
#endif

using namespace Database;

using Command::checkLastMoveIsNotPass;
using Command::parseLegalCoord;
using Command::readPathFromInput;

namespace Command::GomocupProtocol {

std::unique_ptr<Board>        board;
Search::SearchOptions         options;
bool                          GUIMode   = false;
std::atomic_bool              thinking  = false;
std::atomic_bool              exiting   = false;
std::optional<CandidateRange> candRange = std::nullopt;

void sendActionAndUpdateBoard(ActionType action, Pos bestMove)
{
    if (action == ActionType::Move) {
        board->move(options.rule, bestMove);
        std::cout << CoordText {bestMove, board->size(), Config::GeneralCfg.ioCoordMode} << std::endl;
    }
    else if (action == ActionType::Move2) {
        Pos move1 = Search::Engine.main()->rootMoves[0].pv[0];
        Pos move2 = Search::Engine.main()->rootMoves[0].pv[1];
        board->move(options.rule, move1);
        board->move(options.rule, move2);
        std::cout << CoordText {move1, board->size(), Config::GeneralCfg.ioCoordMode} << ' '
                  << CoordText {move2, board->size(), Config::GeneralCfg.ioCoordMode} << std::endl;
    }
    else if (action == ActionType::Swap) {
        std::cout << "SWAP" << std::endl;
    }
    else if (action == ActionType::Swap2PutTwo) {
        if (Search::Engine.isTerminating())
            return;

        options.swapable    = false;
        options.balanceMode = Search::SearchOptions::BALANCE_TWO;
        Search::Engine.startThinking(*board, options, false, []() {
            Pos move1 = Search::Engine.main()->rootMoves[0].pv[0];
            Pos move2 = Search::Engine.main()->rootMoves[0].pv[1];
            std::cout << CoordText {move1, board->size(), Config::GeneralCfg.ioCoordMode} << ' '
                      << CoordText {move2, board->size(), Config::GeneralCfg.ioCoordMode} << std::endl;
        });
    }
}

void think(Board                             &board,
           uint16_t                           multiPV     = 1,
           Search::SearchOptions::BalanceMode balanceMode = Search::SearchOptions::BALANCE_NONE,
           bool                               swapable    = false,
           bool                               disableOpeningQuery = false)
{
    options.multiPV             = multiPV;
    options.balanceMode         = balanceMode;
    options.swapable            = swapable;
    options.disableOpeningQuery = disableOpeningQuery;

    // If threads are pondering, stop them now and wait for them to finish.
    // This must precede the config reload below: loadConfig quiesces the
    // search threads without signaling them to stop.
    if (Search::Engine.ctx.inPonder) {
        Search::Engine.stopThinking();
        Search::Engine.waitForIdle();
    }

    if (Config::GeneralCfg.reloadConfigEachMove)
        loadConfig();

    thinking = true;
    Search::Engine.startThinking(board, options, false, [startTime = now()]() {
        // This callback runs on the main search thread while the protocol
        // thread may be reading commands. Hold protocolMutex over the whole
        // callback: `thinking` only becomes false under the lock, so once the
        // protocol thread observes it false and acquires the lock, the shared
        // options and board are fully settled and any ponder search has been
        // started. This orders the ponder startup before any later stop
        // request or board reallocation issued by a protocol command.
#ifdef MULTI_THREADING
        std::lock_guard<std::mutex> lock(protocolMutex);
#endif

        sendActionAndUpdateBoard(Search::Engine.ctx.resultAction, Search::Engine.ctx.bestMove);
        thinking = false;

        // Subtract used match time
        Time usedTime = now() - startTime;
        if (options.timeLimit && options.matchTime > 0)
            options.timeLeft = std::max(options.timeLeft - usedTime, (Time)0);

        // Start pondering search if needed. The global board is re-fetched
        // under the lock (it cannot be reallocated while we hold it), and no
        // ponder search may start once the protocol loop is exiting, as
        // nothing would stop it. Pondering requires a worker thread: in a
        // single-threaded build the ponder search would run synchronously on
        // the protocol thread and could never receive a stop command.
#ifdef MULTI_THREADING
        if (Search::Engine.ctx.startPonderAfterThinking && !exiting)
            Search::Engine.startThinking(*GomocupProtocol::board, options, true);
#endif
    });
}

void setGUIMode()
{
    MESSAGEL("Rapfi " << getVersionInfo());
    MESSAGEL("INFO MAX_THREAD_NUM 256");
    MESSAGEL("INFO MAX_HASH_SIZE 30");
    GUIMode = true;
    if (Config::GeneralCfg.messageMode == MsgMode::BRIEF)
        Config::GeneralCfg.messageMode = MsgMode::NORMAL;
}

/// Set the searcher memory limit to the given budget minus the reserved amount,
/// flooring the result at the minimal limit of 1 KiB.
void setSearcherMemoryLimit(size_t maxMemSizeKB, size_t memReservedKB)
{
    size_t memLimitKB = maxMemSizeKB <= memReservedKB ? 1 : maxMemSizeKB - memReservedKB;
    Search::Engine.searcher()->setMemoryLimit(memLimitKB);
}

void getOption()
{
    std::string token, str;
    int64_t     val;

    std::cin >> token;

    if (Tuning::TuneMap::tryReadOption(token, std::cin))
        return;

    upperInplace(token);

    if (token == "TIMEOUT_TURN") {
        std::cin >> val;
        options.setTimeControl(val, options.matchTime);
        if (GUIMode && options.matchTime == 100000000 && options.turnTime == 2000000)
            options.timeLimit = false;
    }
    else if (token == "TIMEOUT_MATCH") {
        std::cin >> val;
        options.setTimeControl(options.turnTime, val);
        if (GUIMode && options.matchTime == 100000000 && options.turnTime == 2000000)
            options.timeLimit = false;
    }
    else if (token == "TIME_LEFT") {
        std::cin >> val;
        if (val == 2147483647)  // unlimited match time specificed in the protocol
            options.timeLeft = std::numeric_limits<Time>::max();
        else
            options.timeLeft = val;
    }
    else if (token == "MAX_MEMORY" || token == "HASH_SIZE" /* Yixin-Board Extension*/) {
        std::cin >> val;
        size_t maxMemSizeKB  = val;  // max memory usage in KiB
        size_t memReservedKB = 0;

        if (token == "MAX_MEMORY") {
            memReservedKB = Config::GeneralCfg.memoryReservedMB[options.rule.rule] * 1024;
            if (val == 0) {
                maxMemSizeKB = 350 * 1024;  // use default gomocup max_memory
            }
            else {
                maxMemSizeKB = val >> 10;  // max memory usage in KB
                // Warn if max memory is less than 10MB or reserved memory size
                if (maxMemSizeKB < std::max<size_t>(10240, memReservedKB))
                    ERRORL("Max memory too small, might exceeds memory limits");
            }
        }

        setSearcherMemoryLimit(maxMemSizeKB, memReservedKB);
    }
    else if (token == "RULE") {
        std::cin >> val;

        Rule prevRule = options.rule;
        switch (val) {
        case 0: options.rule = {Rule::FREESTYLE, GameRule::FREEOPEN}; break;
        case 1: options.rule = {Rule::STANDARD, GameRule::FREEOPEN}; break;
        case 2:  // Yixin-Board Extension
        case 4: options.rule = {Rule::RENJU, GameRule::FREEOPEN}; break;
        case 5: options.rule = {Rule::FREESTYLE, GameRule::SWAP1}; break;
        case 6: options.rule = {Rule::FREESTYLE, GameRule::SWAP2}; break;
        default:
            ERRORL("Unknown rule id: " << val << ". Rule is reset to freestyle...");
            options.rule = {Rule::FREESTYLE, GameRule::FREEOPEN};
        }

        // Resize TT if memory reserved is different for this rule.
        if (Config::GeneralCfg.memoryReservedMB[prevRule] != Config::GeneralCfg.memoryReservedMB[options.rule.rule]) {
            size_t maxMemSizeKB = Search::Engine.searcher()->getMemoryLimit()
                                  + Config::GeneralCfg.memoryReservedMB[prevRule] * 1024;
            setSearcherMemoryLimit(maxMemSizeKB,
                                   Config::GeneralCfg.memoryReservedMB[options.rule.rule] * 1024);
        }

        // Clear TT if rule is changed
        if (board && prevRule != options.rule) {
            board->newGame(options.rule);
            Search::Engine.clear(true);
        }
    }
    /////////////////////////////////////////////////
    // Yixin - Board Extension
    /////////////////////////////////////////////////
    else if (token == "MAX_DEPTH") {
        std::cin >> options.maxDepth;
    }
    else if (token == "START_DEPTH") {
        std::cin >> options.startDepth;
    }
    else if (token == "MAX_NODE") {
        std::cin >> options.maxNodes;
        if (options.maxNodes == ULLONG_MAX)
            options.maxNodes = 0;
    }
    else if (token == "SHOW_DETAIL") {
        std::cin >> val;
        switch (val) {
        default: MESSAGEL("Unknown show_detail, setting to 0."); [[fallthrough]];
        case 0: options.infoMode = Search::SearchOptions::INFO_NONE; break;
        case 1: options.infoMode = Search::SearchOptions::INFO_REALTIME; break;
        case 2: options.infoMode = Search::SearchOptions::INFO_DETAIL; break;
        case 3: options.infoMode = Search::SearchOptions::INFO_REALTIME_AND_DETAIL; break;
        }
    }
    else if (token == "CAUTION_FACTOR") {
        std::cin >> val;

        auto prevCandRange = candRange;

        switch (val) {
        case 0: candRange = CandidateRange::SQUARE2; break;
        default: MESSAGEL("Unknown caution_factor, setting to 1."); [[fallthrough]];
        case 1: candRange = CandidateRange::SQUARE2_LINE3; break;
        case 2: candRange = CandidateRange::SQUARE3; break;
        case 3: candRange = CandidateRange::SQUARE3_LINE4; break;
        case 4: candRange = CandidateRange::SQUARE4; break;
        case 5: candRange = CandidateRange::FULL_BOARD; break;
        }

        // Make a new board from scratch to match the new candidate range
        if (board && (!prevCandRange.has_value() || *prevCandRange != *candRange)) {
            std::vector<Pos> tempPosition;
            for (int i = 0; i < board->ply(); i++)
                tempPosition.push_back(board->getHistoryMove(i));

            board = std::make_unique<Board>(board->size(), *candRange);
            board->newGame(options.rule);
            for (Pos p : tempPosition)
                board->move(options.rule, p);

            // Clear TT in case of some mistaken win/loss records
            Search::Engine.clear(true);
        }
    }
    else if (token == "THREAD_NUM") {
        std::cin >> val;
        val = std::max<int64_t>(val, 1);
        if (Search::Engine.size() != val) {
            Search::Engine.setNumThreads(val);
        }
    }
    else if (token == "PONDERING") {
        std::cin >> val;
        options.pondering = val == 1;
    }
    // Protocol-required no-ops: these Gomocup/Yixin-Board options are accepted
    // (their value must be consumed from the stream) but deliberately ignored.
    else if (token == "GAME_TYPE" || token == "TIME_INCREMENT" || token == "THREAD_SPLIT_DEPTH"
             || token == "CHECKMATE" || token == "NBESTSYM" || token == "VCTHREAD") {
        std::cin >> val;
    }
    else if (token == "FOLDER") {
        std::cin >> str;  // persistent folder path, deliberately ignored
    }
    else if (token == "USEDATABASE") {
        std::cin >> val;
        if (val == 1)
            Search::Engine.setupDatabase(Database::createDBStorage(Database::DatabaseCfg));
        else
            Search::Engine.setupDatabase(nullptr);
    }
    /////////////////////////////////////////////////
    // Other Extension
    /////////////////////////////////////////////////
    else if (token == "DATABASE_READONLY") {
        std::cin >> val;
        Database::DatabaseCfg.search.readonlyMode = val == 1;
    }
    else if (token == "SWAPABLE") {
        std::cin >> val;
        options.swapable = val == 1;
    }
    else if (token == "STRENGTH") {
        std::cin >> options.strengthLevel;
    }
    else if (token == "MAX_MOVES") {
        std::cin >> val;
        if (val <= 0)
            val = INT32_MAX;

        if (val != options.maxMoves) {
            options.maxMoves = val;

            // We need to clear TT in case of mistaken results stored
            Search::Engine.clear(true);
        }
    }
    else if (token == "DRAW_RESULT") {
        std::cin >> val;

        auto prevDrawResult = options.drawResult;

        switch (val) {
        default: MESSAGEL("Unknown draw_result, setting to 0."); [[fallthrough]];
        case 0: options.drawResult = Search::SearchOptions::RES_DRAW; break;
        case 1: options.drawResult = Search::SearchOptions::RES_BLACK_WIN; break;
        case 2: options.drawResult = Search::SearchOptions::RES_WHITE_WIN; break;
        }

        if (prevDrawResult != options.drawResult) {
            // We need to clear TT in case of mistaken results stored
            Search::Engine.clear(true);
        }
    }
    else if (token == "EVALUATOR_DRAW_BLACK_WINRATE") {
        std::cin >> Evaluation::EvalCfg.drawBlackWinRate;
        Evaluation::EvalCfg.drawBlackWinRate =
            std::clamp(Evaluation::EvalCfg.drawBlackWinRate, 0.0f, 1.0f);
    }
    else if (token == "EVALUATOR_DRAW_RATIO") {
        std::cin >> Evaluation::EvalCfg.drawRatio;
        Evaluation::EvalCfg.drawRatio = std::clamp(Evaluation::EvalCfg.drawRatio, 0.0f, 1.0f);
    }
    else if (token == "SEARCH_TYPE") {
        std::cin >> str;
        Search::Engine.setupSearcher(Search::createSearcher(str));
        Search::Engine.clear(true);
    }
    else {
        MESSAGEL("Unknown Info Parameter: " << token);
    }
}

void clearHash()
{
    Search::Engine.clear(true);
    MESSAGEL("Transposition table cleared.");
}

void showHashUsage()
{
    int hashfull_permill = Search::TT.hashUsage();
    MESSAGEL("Transposition table full: " << hashfull_permill / 10 << "." << hashfull_permill % 10
                                          << "%");
}

void dumpHash()
{
    auto          path = readPathFromInput();
    std::ofstream hashout(path, std::ios_base::binary);
    if (hashout.is_open()) {
        Search::TT.dump(hashout);
        MESSAGEL("Transposition table dumped: " << pathToConsoleString(path));
    }
    else
        MESSAGEL("Failed to open file: " << pathToConsoleString(path));
}

void loadHash()
{
    auto          path = readPathFromInput();
    std::ifstream hashin(path, std::ios_base::binary);
    if (hashin.is_open() && Search::TT.load(hashin))
        MESSAGEL("Transposition table loaded successfully from " << pathToConsoleString(path));
    else
        MESSAGEL("Transposition table loaded failed, "
                 "please check if the file is correct.");
}

void reloadConfig()
{
    configPath          = readPathFromInput();
    allowInternalConfig = configPath.empty();
    if (allowInternalConfig)
        MESSAGEL("No external config specified, reload internal config.");

    if (!loadConfig())
        ERRORL("Failed to load config. Please check if config file is correct.");
}

/// Builds the context for the database protocol handlers in dbcommand.cpp.
Command::DBCommandContext dbCommandContext()
{
    return {std::cin, board.get(), options, Search::Engine.dbStorage()};
}

void setDatabase()
{
    auto databasePath = readPathFromInput();
    if (!databasePath.empty() && !Database::DatabaseCfg.type.empty()) {
        Database::DatabaseCfg.url     = databasePath.u8string();
        auto newDatabaseStorage = Database::createDBStorage(Database::DatabaseCfg);
        if (newDatabaseStorage)
            Search::Engine.setupDatabase(std::move(newDatabaseStorage));
    }
}

void restart()
{
    board->newGame(options.rule);
    Search::Engine.clear(false);
    std::cout << "OK" << std::endl;
}

void start()
{
    int boardSize;
    std::cin >> boardSize;
    if (boardSize < 5 || boardSize > MAX_BOARD_SIZE) {
        ERRORL("Unsupported board size!");
        return;
    }

    if (!board || boardSize != board->size()) {
        auto candidateRange = candRange.value_or(Config::GeneralCfg.defaultCandidateRange);
        board               = std::make_unique<Board>(boardSize, candidateRange);
    }

    restart();
}

void rectStart()
{
    int  x, y;
    char comma;
    std::cin >> x >> comma >> y;
    ERRORL("Rectangular board is not supported.");
}

void takeBack()
{
    int  x, y;
    char comma;
    std::cin >> x >> comma >> y;

    if (board->ply() > 0) {
        board->undo(options.rule);
        std::cout << "OK" << std::endl;
    }
    else
        ERRORL("Board is empty now.");
}

void begin()
{
    if (board->ply() != 0)
        ERRORL("Board is not empty.");
    else
        think(*board);
}

void turn()
{
    auto pos = parseLegalCoord(std::cin, *board);
    if (!pos.has_value())
        return;

    options.multiPV     = 1;
    options.balanceMode = Search::SearchOptions::BALANCE_NONE;
    board->move(options.rule, *pos);
    think(*board);
}

void getPosition(bool startThink)
{
    board->newGame(options.rule);
    options.multiPV     = 1;
    options.balanceMode = Search::SearchOptions::BALANCE_NONE;

    // Read position sequence
    enum SideFlag { SELF = 1, OPPO = 2, WALL = 3 };
    std::vector<std::pair<Pos, SideFlag>> position;

    while (true) {
        std::string coordStr;
        std::cin >> coordStr;
        upperInplace(coordStr);

        if (coordStr == "DONE" || std::cin.eof())
            break;

        std::optional<Pos> pos;
        int                color = -1;
        {
            std::stringstream ss;
            char              comma;
            ss << coordStr;
            pos = parseLegalCoord(ss, *board);
            ss >> comma >> color;
        }

        if (pos.has_value()) {
            if (color == SELF || color == OPPO || color == WALL && *pos != Pos::NONE)
                position.emplace_back(*pos, static_cast<SideFlag>(color));
            else
                ERRORL("Color is not a valid value, must be one of [1, 2, 3].");
        }
    }

    // The first move (either real move or pass) is always considered as BLACK
    Color selfColor = BLACK;
    for (auto [pos, side] : position) {
        if (side != WALL) {
            selfColor = side == SELF ? BLACK : WHITE;
            break;
        }
    }

    // Put stones on board
    for (auto [pos, side] : position) {
        if (side == WALL)  // Currently wall is not supported
            continue;

        // Cap the total number of moves: one entry may add an extra pass plus
        // the move itself, and the per-ply board state arrays hold at most
        // cellCount()*2 moves.
        if (board->ply() + 2 > board->cellCount() * 2) {
            ERRORL("Too many moves in the position.");
            return;
        }

        // Make sure current side to move correspond to the input side
        // If not, we add an extra PASS move to flip the side
        if (side == SELF && board->sideToMove() != selfColor
            || side == OPPO && board->sideToMove() != ~selfColor) {
            if (checkLastMoveIsNotPass(*board))
                return;

            board->move(options.rule, Pos::PASS);
        }

        if (pos == Pos::PASS && checkLastMoveIsNotPass(*board))
            return;

        // The read loop validated coordinates against an empty board, so
        // duplicated coordinates can only be rejected here, against the
        // actual position built so far.
        if (!board->isLegal(pos)) {
            ERRORL("Coord is not valid or empty.");
            return;
        }

        board->move(options.rule, pos);
    }

    // Start thinking if needed
    if (startThink)
        think(*board);
}

void getBlock(bool remove = false)
{
    int               x, y;
    char              comma;
    std::string       coordStr;
    std::stringstream ss;

    while (true) {
        std::cin >> coordStr;
        upperInplace(coordStr);

        if (coordStr == "DONE" || std::cin.eof())
            break;

        // Reset the stream to hold only this token, so that a previous bad
        // token can not break the parse of the following coordinates
        ss.str(coordStr);
        ss.clear();
        if (!(ss >> x >> comma >> y)) {
            ERRORL("Block coord is a pass or invalid.");
            continue;
        }

        Pos pos = inputCoordConvert(x, y, board->size(), Config::GeneralCfg.ioCoordMode);
        if (!board->isInBoard(pos) || pos == Pos::PASS) {
            ERRORL("Block coord is a pass or invalid.");
            continue;
        }

        bool alreadyInList = std::count(options.blockMoves.begin(), options.blockMoves.end(), pos);
        if (!remove) {
            if (!alreadyInList)
                options.blockMoves.push_back(pos);
        }
        else if (alreadyInList) {
            options.blockMoves.erase(
                std::remove(options.blockMoves.begin(), options.blockMoves.end(), pos),
                options.blockMoves.end());
        }
    }
}

void nbest()
{
    std::cin >> options.multiPV;
    think(*board, std::max<uint16_t>(options.multiPV, 1));
}

void showForbid()
{
    std::cout << "FORBID ";
    if (board->sideToMove() == BLACK) {
        FOR_EVERY_EMPTY_POS(board, pos)
        {
            if (board->checkForbiddenPoint(pos)) {
                auto [cx, cy] = outputCoordConvert(pos, board->size(), Config::GeneralCfg.ioCoordMode);
                std::cout << std::setfill('0') << std::setw(2) << cx << std::setfill('0')
                          << std::setw(2) << cy;
            }
        }
    }
    std::cout << '.' << std::endl;
}

void balance(Search::SearchOptions::BalanceMode mode)
{
    std::cin >> options.balanceBias;

    // Revert bias for balance two mode
    if (mode == Search::SearchOptions::BALANCE_TWO)
        options.balanceBias = -options.balanceBias;

    if (board->ply() == 0)
        Opening::expandCandidateHalfBoard(*board);

    think(*board, 1, mode, false, true);
}

void searchDefend()
{
    options.multiPV = board->movesLeft();
    think(*board, std::max<uint16_t>(options.multiPV, 1));
}

void swap2board()
{
    options.rule.opRule = GameRule::SWAP2;
    getPosition(false);

    // For empty board, we manually put some openings
    if (board->ply() == 0) {
        Pos        opening[3];
        ActionType action;
        if (Opening::probeOpening(*board, options.rule, action, opening[0])) {
            opening[0] = board->getRecentMove(2);
            opening[1] = board->getRecentMove(1);
            opening[2] = board->getRecentMove(0);
        }
        else {
            MESSAGEL("No available opening is in database!");
            opening[0] = Pos(0, 0);
            opening[1] = Pos(2, 1);
            opening[2] = Pos(2, 4);
        }

        std::cout << CoordText {opening[0], board->size(), Config::GeneralCfg.ioCoordMode} << ' '
                  << CoordText {opening[1], board->size(), Config::GeneralCfg.ioCoordMode} << ' '
                  << CoordText {opening[2], board->size(), Config::GeneralCfg.ioCoordMode} << std::endl;
    }
    else {
        think(*board, 1, Search::SearchOptions::BALANCE_NONE, true);
    }
}

void traceBoard()
{
    Search::Engine.waitForIdle();
    Search::Engine.ctx.options = options;
    Search::Engine.main()->setBoardAndEvaluator(*board);

    std::string traceInfo  = Search::Engine.main()->board->trace(options.rule);
    auto        traceLines = split(traceInfo, "\n");
    for (const auto &line : traceLines) {
        MESSAGEL(line);
    }
}

void traceSearch()
{
    Search::Engine.waitForIdle();
    Search::Engine.ctx.options = options;
    Search::Engine.main()->setBoardAndEvaluator(*board);
    Board &board = *Search::Engine.main()->board;

    // Legal moves
    Search::MovePicker movePicker(options.rule,
                                  board,
                                  Search::MovePicker::ExtraArgs<Search::MovePicker::ROOT> {});
    std::vector<Pos>   moveList;
    while (Pos move = movePicker())
        moveList.push_back(move);
    MESSAGEL("Legal Moves[" << moveList.size() << "]: " << MovesText {moveList});

    Opening::filterSymmetryMoves(board, moveList);
    MESSAGEL("Root Moves(exclude symmetry)[" << moveList.size() << "]: " << MovesText {moveList});

    // Static evaluation (black point of view)
    Value eval    = Evaluation::evaluate(board, options.rule);
    Value evalPOB = board.sideToMove() == BLACK ? eval : -eval;
    MESSAGEL("Static Eval[Black]: " << evalPOB << " (WDL " << std::fixed << std::setprecision(2)
                                    << (Evaluation::valueToWinRate(evalPOB) * 100.0f) << ", SF "
                                    << std::setprecision(2) << Evaluation::ScalingFactor << ")");

    if (board.evaluator()) {
        auto  v     = board.evaluator()->evaluateValue(board);
        Value value = board.sideToMove() == BLACK ? v.value() : -v.value();
        float win   = board.sideToMove() == BLACK ? v.win() : v.loss();
        float loss  = board.sideToMove() == BLACK ? v.loss() : v.win();
        MESSAGEL("Evaluator Eval[Black]: " << value << " WR: " << std::setprecision(4) << win
                                           << " LR: " << loss << " DR: " << v.draw());
    }

    // Hash entry
    Value ttValue;
    Value ttEval;
    bool  ttIsPv;
    Bound ttBound;
    Pos   ttMove;
    int   ttDepth;
    if (Search::TT
            .probe(board.zobristKey(), ttValue, ttEval, ttIsPv, ttBound, ttMove, ttDepth, 0)) {
        const char *BoundType[] = {"x", "<", ">", "="};

        MESSAGEL("TTEntry: " << (ttIsPv ? "(pv)" : ""));
        MESSAGEL("  depth: " << ttDepth);
        MESSAGEL("  value: (" << BoundType[ttBound] << ") " << ttValue);
        MESSAGEL("  eval: " << ttEval);
        MESSAGEL("  bestmove: " << ttMove);
    }
    else
        MESSAGEL("No TTEntry found.");
}

void exportModel()
{
    auto          modelPath = readPathFromInput();
    std::ofstream modelFile(modelPath, std::ios::binary);
    if (!modelFile.is_open())
        ERRORL("Unable to open weight file: " << modelPath);
    else
        Config::exportModel(modelFile);
}

void loadModel()
{
    auto modelPath = readPathFromInput();
    loadModelFromFile(modelPath);
}

/// Enter protocol loop once and fetch and execute one command from stdin.
/// @return True if program should exit now.
bool runProtocol()
{
    // Read the command from stdin in a blocking way
    std::string cmd;
    std::cin >> cmd;

    // Stop the protocol loop when reaching EOF
    // We do not stop thinking here, but let the outer loop to wait for its finish.
    if (std::cin.eof())
        return true;

    // We assume the command is in uppercase, but also support lowercase
    upperInplace(cmd);

    auto CheckBoardOK = [&](auto f) {
        if (!board)
            ERRORL("No game has been started.");
        else
            f();
    };

    // clang-format off
    if (cmd == "END")                          return Search::Engine.stopThinking(), true;
    else if (cmd == "STOP" || cmd == "YXSTOP") {
        // Raw stop first for minimal latency mid-search. A stop that races the
        // finish callback loses: the callback's nested ponder startThinking
        // resets the terminate flag after us. Acquiring protocolMutex
        // serializes behind any in-flight callback, so the second stop lands
        // on a just-started ponder; when nothing is running it is a no-op
        // cleared by the next startThinking.
        Search::Engine.stopThinking();
#ifdef MULTI_THREADING
        std::lock_guard<std::mutex> stopLock(protocolMutex);
        Search::Engine.stopThinking();
#endif
        return false;
    }
    // Any other command is dropped while thinking; discard the rest of its
    // line as well, so its argument tokens are not parsed as commands later.
    else if (thinking)
        return std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'), false;

#ifdef MULTI_THREADING
    std::lock_guard<std::mutex> lock(protocolMutex);
#endif

    // Stop pondering first for commands that may modify the board state
    if (cmd != "TURN"
     && cmd != "YXSHOWFORBID"
     && cmd != "YXBLOCKRESET"
     && cmd != "YXQUERYDATABASEALL"
     && cmd != "YXQUERYDATABASEONE"
     && cmd != "YXQUERYDATABASETEXT"
     && cmd != "YXQUERYDATABASEALLT")      Search::Engine.stopThinking();

    if (cmd == "ABOUT")                    std::cout << getEngineInfo() << std::endl;
    else if (cmd == "START")               start();
    else if (cmd == "RECTSTART")           rectStart();
    else if (cmd == "INFO")                getOption();
    else if (cmd == "YXSHOWINFO")          setGUIMode();
    else if (cmd == "YXHASHCLEAR")         clearHash();
    else if (cmd == "YXSHOWHASHUSAGE")     showHashUsage();
    else if (cmd == "YXHASHDUMP")          dumpHash();
    else if (cmd == "YXHASHLOAD")          loadHash();
    else if (cmd == "YXSETDATABASE")       setDatabase();
    else if (cmd == "YXSAVEDATABASE")      { auto ctx = dbCommandContext(); saveDatabase(ctx); }
    else if (cmd == "YXDBTOTXTALL")        { auto ctx = dbCommandContext(); databaseToTxt(ctx, false); }
    else if (cmd == "YXDBTOTXT")           { auto ctx = dbCommandContext(); databaseToTxt(ctx, true); }
    else if (cmd == "YXLIBTODB")           { auto ctx = dbCommandContext(); libToDatabase(ctx); }
    else if (cmd == "YXDBTOLIB")           CheckBoardOK([] { auto ctx = dbCommandContext(); databaseToLib(ctx); });
    else if (cmd == "RELOADCONFIG")        reloadConfig();
    else if (cmd == "LOADMODEL")           loadModel();
    else if (cmd == "EXPORTMODEL")         exportModel();
    else if (cmd == "BENCH")               benchmark();
    else if (cmd == "RESTART")             CheckBoardOK(restart);
    else if (cmd == "TAKEBACK")            CheckBoardOK(takeBack);
    else if (cmd == "BEGIN")               CheckBoardOK(begin);
    else if (cmd == "TURN")                CheckBoardOK(turn);
    else if (cmd == "BOARD")               CheckBoardOK([] { getPosition(true); });
    else if (cmd == "YXBOARD")             CheckBoardOK([] { getPosition(false); });
    else if (cmd == "YXBLOCK")             CheckBoardOK([] { getBlock(false); });
    else if (cmd == "YXBLOCKUNDO")         CheckBoardOK([] { getBlock(true); });
    else if (cmd == "YXBLOCKRESET")        CheckBoardOK([] { options.blockMoves.clear(); });
    else if (cmd == "YXNBEST")             CheckBoardOK(nbest);
    else if (cmd == "YXSHOWFORBID")        CheckBoardOK(showForbid);
    else if (cmd == "YXBALANCEONE")        CheckBoardOK([] { balance(Search::SearchOptions::BALANCE_ONE); });
    else if (cmd == "YXBALANCETWO")        CheckBoardOK([] { balance(Search::SearchOptions::BALANCE_TWO); });
    else if (cmd == "YXQUERYDATABASEALL")  CheckBoardOK([] { auto ctx = dbCommandContext(); queryDatabaseAll(ctx, true); });
    else if (cmd == "YXQUERYDATABASEONE")  CheckBoardOK([] { auto ctx = dbCommandContext(); queryDatabaseOne(ctx, true); });
    else if (cmd == "YXQUERYDATABASETEXT") CheckBoardOK([] { auto ctx = dbCommandContext(); queryDatabaseText(ctx, true); });
    else if (cmd == "YXQUERYDATABASEALLT") CheckBoardOK([] { auto ctx = dbCommandContext(); queryDatabaseAll(ctx, true); queryDatabaseText(ctx, false); });
    else if (cmd == "YXEDITTVDDATABASE")   CheckBoardOK([] { auto ctx = dbCommandContext(); editDatabaseTVD(ctx); });
    else if (cmd == "YXEDITTEXTDATABASE")  CheckBoardOK([] { auto ctx = dbCommandContext(); editDatabaseText(ctx); });
    else if (cmd == "YXEDITLABELDATABASE") CheckBoardOK([] { auto ctx = dbCommandContext(); editDatabaseBoardLabel(ctx); });
    else if (cmd == "YXDELETEDATABASEONE") CheckBoardOK([] { auto ctx = dbCommandContext(); deleteDatabaseOne(ctx, true); });
    else if (cmd == "YXDELETEDATABASEALL") CheckBoardOK([] { auto ctx = dbCommandContext(); deleteDatabaseAll(ctx, true); });
    else if (cmd == "YXSEARCHDEFEND")      CheckBoardOK(searchDefend);
    else if (cmd == "YXDBSPLIT")           CheckBoardOK([] { auto ctx = dbCommandContext(); splitDatabase(ctx); });
    else if (cmd == "YXDBMERGE")           CheckBoardOK([] { auto ctx = dbCommandContext(); mergeDatabase(ctx); });
    else if (cmd == "SWAP2BOARD")          CheckBoardOK(swap2board);
    else if (cmd == "TRACEBOARD")          CheckBoardOK(traceBoard);
    else if (cmd == "TRACESEARCH")         CheckBoardOK(traceSearch);
    else if (!GUIMode)                     ERRORL("Unknown command: " << cmd);
    // clang-format on

    return false;
}

}  // namespace Command::GomocupProtocol

#ifdef __EMSCRIPTEN__
    #include <emscripten.h>

    #ifdef MULTI_THREADING
        #include <emscripten/atomic.h>
static uint32_t loop_ready = 0;  // count variable for atomic wait
    #endif

/// Entry point of one gomocup protocol iter for WASM build
extern "C" void gomocupLoopOnce()
{
    #ifdef MULTI_THREADING
    emscripten_atomic_add_u32(&loop_ready, 1);
    emscripten_atomic_notify(&loop_ready, 1);
    #else
    if (Command::GomocupProtocol::runProtocol())
        emscripten_force_exit(EXIT_SUCCESS);
    #endif
}
#endif

/// Wrap around runProtocol(), looping until exit condition is met.
/// This will only return after all searching threads have ended.
void Command::gomocupLoop()
{
    // Init tuning parameter table
    Tuning::TuneMap::init();

    for (;;) {
#ifdef __EMSCRIPTEN__
    #ifdef MULTI_THREADING
        while (emscripten_atomic_load_u32(&loop_ready) == 0)
            emscripten_atomic_wait_u32(&loop_ready, 0, -1);
        emscripten_atomic_sub_u32(&loop_ready, 1);
    #else
        // We do not run infinite loop in single thread build, instead we manually
        // call gomocupLoopOnce() for each command to avoid hanging the main thread.
        return;
    #endif
#endif

        // Run the protocol loop for one iter
        if (GomocupProtocol::runProtocol())
            break;

#ifdef MULTI_THREADING
        // For multi-threading build, yield before reading the next
        // command to avoid possible busy waiting. Most of time reading
        // from std::cin would block so this may not necessary.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
    }

    // Forbid the search-finish callback from starting a new ponder search, and
    // stop a running one (a ponder search never stops by itself), otherwise the
    // wait below would never return. A normal search is left to finish so that
    // its best move is still printed.
    {
#ifdef MULTI_THREADING
        std::lock_guard<std::mutex> lock(protocolMutex);
#endif
        GomocupProtocol::exiting = true;
        if (Search::Engine.ctx.inPonder)
            Search::Engine.stopThinking();
    }

    // If there is any thread still running, wait until they exited.
    Search::Engine.waitForIdle();
    Search::Engine.setNumThreads(0);

#ifdef __EMSCRIPTEN__
    emscripten_force_exit(EXIT_SUCCESS);
#endif
}
