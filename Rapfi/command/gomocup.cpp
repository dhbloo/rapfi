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
#include "../database/dbclient.h"
#include "../database/dbutils.h"
#include "../database/yxdbstorage.h"
#include "../eval/eval.h"
#include "../eval/evaluator.h"
#include "../game/board.h"
#include "../search/ab/searcher.h"
#include "../search/hashtable.h"
#include "../search/mcts/searcher.h"
#include "../search/movepick.h"
#include "../search/opening.h"
#include "../search/searchthread.h"
#include "../tuning/tunemap.h"
#include "command.h"

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

namespace {

std::filesystem::path readPathFromInput(std::istream &in = std::cin)
{
    std::string path;
    in >> std::ws;
    std::getline(in, path);
    trimInplace(path);

    return pathFromConsoleString(path);
}

bool checkLastMoveIsNotPass(Board &board)
{
    if (board.passMoveCount() > 0 && board.getLastMove() == Pos::PASS) {
        ERRORL("Consecutive pass is not supported. "
               "There must be one non-pass move between two pass moves.");
        return true;
    }
    return false;
}

/// Parse a coord in form 'x,y' from in stream and check if the pos is legal.
/// Legal means the pos must be an empty cell or an non-consecutive pass move.
std::optional<Pos> parseLegalCoord(std::istream &is, Board &board)
{
    int  x = -2, y = -2;
    char comma;
    is >> x >> comma >> y;

    Pos pos = inputCoordConvert(x, y, board.size());
    if (board.isLegal(pos)) {
        if (checkLastMoveIsNotPass(board))
            return std::nullopt;
        return pos;
    }
    else {
        ERRORL("Coord is not valid or empty.");
        return std::nullopt;
    }
}

}  // namespace

namespace Command::GomocupProtocol {

std::unique_ptr<Board>        board;
Search::SearchOptions         options;
bool                          GUIMode   = false;
std::atomic_bool              thinking  = false;
std::optional<CandidateRange> candRange = std::nullopt;

void sendActionAndUpdateBoard(ActionType action, Pos bestMove)
{
    if (action == ActionType::Move) {
        board->move(options.rule, bestMove);
        std::cout << outputCoordXConvert(bestMove, board->size()) << ','
                  << outputCoordYConvert(bestMove, board->size()) << std::endl;
    }
    else if (action == ActionType::Move2) {
        Pos move1 = Search::Threads.main()->rootMoves[0].pv[0];
        Pos move2 = Search::Threads.main()->rootMoves[0].pv[1];
        board->move(options.rule, move1);
        board->move(options.rule, move2);
        std::cout << outputCoordXConvert(move1, board->size()) << ','
                  << outputCoordYConvert(move1, board->size()) << ' '
                  << outputCoordXConvert(move2, board->size()) << ','
                  << outputCoordYConvert(move2, board->size()) << std::endl;
    }
    else if (action == ActionType::Swap) {
        std::cout << "SWAP" << std::endl;
    }
    else if (action == ActionType::Swap2PutTwo) {
        if (Search::Threads.isTerminating())
            return;

        options.swapable    = false;
        options.balanceMode = Search::SearchOptions::BALANCE_TWO;
        Search::Threads.startThinking(*board, options, false, []() {
            Pos move1 = Search::Threads.main()->rootMoves[0].pv[0];
            Pos move2 = Search::Threads.main()->rootMoves[0].pv[1];
            std::cout << outputCoordXConvert(move1, board->size()) << ','
                      << outputCoordYConvert(move1, board->size()) << ' '
                      << outputCoordXConvert(move2, board->size()) << ','
                      << outputCoordYConvert(move2, board->size()) << std::endl;
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

    if (Config::ReloadConfigEachMove)
        loadConfig();

    // If threads are pondering, stop them now and wait for them to finish
    if (Search::Threads.main()->inPonder) {
        Search::Threads.stopThinking();
        Search::Threads.waitForIdle();
    }

    thinking = true;
    Search::Threads.startThinking(board, options, false, [&, startTime = now()]() {
        {
#ifdef MULTI_THREADING
            std::lock_guard<std::mutex> lock(protocolMutex);
#endif

            sendActionAndUpdateBoard(Search::Threads.main()->resultAction,
                                     Search::Threads.main()->bestMove);
            thinking = false;
        }

        // Subtract used match time
        Time usedTime = now() - startTime;
        if (options.timeLimit && options.matchTime > 0)
            options.timeLeft = std::max(options.timeLeft - usedTime, (Time)0);

        // Start pondering search if needed
        if (Search::Threads.main()->startPonderAfterThinking)
            Search::Threads.startThinking(board, options, true);
    });
}

void setGUIMode()
{
    MESSAGEL("Rapfi " << getVersionInfo());
    MESSAGEL("INFO MAX_THREAD_NUM 256");
    MESSAGEL("INFO MAX_HASH_SIZE 30");
    GUIMode = true;
    if (Config::MessageMode == MsgMode::BRIEF)
        Config::MessageMode = MsgMode::NORMAL;
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
            memReservedKB = Config::MemoryReservedMB[options.rule.rule] * 1024;
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

        // minimal hash size value is 1 KB
        size_t memLimitKB = maxMemSizeKB <= memReservedKB ? 1 : maxMemSizeKB - memReservedKB;
        Search::Threads.searcher()->setMemoryLimit(memLimitKB);
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
        if (Config::MemoryReservedMB[prevRule] != Config::MemoryReservedMB[options.rule.rule]) {
            size_t maxMemSizeKB = Search::Threads.searcher()->getMemoryLimit()
                                  + Config::MemoryReservedMB[prevRule] * 1024;
            size_t memReservedKB = Config::MemoryReservedMB[options.rule.rule] * 1024;
            size_t memLimitKB    = maxMemSizeKB <= memReservedKB ? 1 : maxMemSizeKB - memReservedKB;
            Search::Threads.searcher()->setMemoryLimit(memLimitKB);
        }

        // Clear TT if rule is changed
        if (board && prevRule != options.rule) {
            board->newGame(options.rule);
            Search::Threads.clear(true);
        }
    }
    else if (token == "GAME_TYPE") {
        std::cin >> val;
    }
    else if (token == "FOLDER") {
        std::cin >> str;
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
    else if (token == "TIME_INCREMENT") {
        std::cin >> val;
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
            Search::Threads.clear(true);
        }
    }
    else if (token == "THREAD_NUM") {
        std::cin >> val;
        val = std::max<int64_t>(val, 1);
        if (Search::Threads.size() != val) {
            Search::Threads.setNumThreads(val);
        }
    }
    else if (token == "THREAD_SPLIT_DEPTH") {
        std::cin >> val;
    }
    else if (token == "PONDERING") {
        std::cin >> val;
        options.pondering = val == 1;
    }
    else if (token == "CHECKMATE") {
        std::cin >> val;
    }
    else if (token == "NBESTSYM") {
        std::cin >> val;
    }
    else if (token == "VCTHREAD") {
        std::cin >> val;
    }
    else if (token == "USEDATABASE") {
        std::cin >> val;
        if (val == 1)
            Search::Threads.setupDatabase(Config::createDefaultDBStorage());
        else
            Search::Threads.setupDatabase(nullptr);
    }
    /////////////////////////////////////////////////
    // Other Extension
    /////////////////////////////////////////////////
    else if (token == "DATABASE_READONLY") {
        std::cin >> val;
        Config::DatabaseReadonlyMode = val == 1;
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
            Search::Threads.clear(true);
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
            Search::Threads.clear(true);
        }
    }
    else if (token == "EVALUATOR_DRAW_BLACK_WINRATE") {
        std::cin >> Config::EvaluatorDrawBlackWinRate;
        Config::EvaluatorDrawBlackWinRate =
            std::clamp(Config::EvaluatorDrawBlackWinRate, 0.0f, 1.0f);
    }
    else if (token == "EVALUATOR_DRAW_RATIO") {
        std::cin >> Config::EvaluatorDrawRatio;
        Config::EvaluatorDrawRatio = std::clamp(Config::EvaluatorDrawRatio, 0.0f, 1.0f);
    }
    else if (token == "SEARCH_TYPE") {
        std::cin >> str;
        Search::Threads.setupSearcher(Config::createSearcher(str));
        Search::Threads.clear(true);
    }
    else {
        MESSAGEL("Unknown Info Parameter: " << token);
    }
}

void clearHash()
{
    Search::Threads.clear(true);
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

void setDatabase()
{
    auto databasePath = readPathFromInput();
    if (!databasePath.empty() && !Config::DatabaseType.empty()) {
        Config::DatabaseURL     = databasePath.u8string();
        auto newDatabaseStorage = Config::createDefaultDBStorage();
        if (newDatabaseStorage)
            Search::Threads.setupDatabase(std::move(newDatabaseStorage));
    }
}

void saveDatabase()
{
    if (Search::Threads.dbStorage()) {
        auto startTime = now();
        Search::Threads.dbStorage()->flush();
        auto endTime = now();
        MESSAGEL("Saved database file using " << (endTime - startTime) << " ms.");
    }
}

void databaseToTxt(bool currentBoardSizeAndRule)
{
    auto txtPath = readPathFromInput();
    if (Search::Threads.dbStorage()) {
        std::ofstream                                        txtout(txtPath);
        std::function<bool(const DBKey &, const DBRecord &)> filter = nullptr;
        if (currentBoardSizeAndRule)
            filter = [&](const DBKey &key, const DBRecord &record) -> bool {
                return key.boardHeight == board->size() && key.boardWidth == board->size()
                       && key.rule == options.rule.rule;
            };
        ::Database::databaseToCSVFile(*Search::Threads.dbStorage(), txtout, filter);
        MESSAGEL("Wrote " << (currentBoardSizeAndRule ? "(current boardsize and rule)" : "(all)")
                          << " database to csv-format text file " << pathToConsoleString(txtPath));
    }
}

void libToDatabase()
{
    auto libPath = readPathFromInput();
    if (Search::Threads.dbStorage()) {
        MESSAGEL("Importing from lib file " << pathToConsoleString(libPath)
                                            << ", this might take a while...");
        auto          startTime = now();
        std::ifstream libStream(libPath, std::ios::binary);
        size_t        writeCount = ::Database::importLibToDatabase(*Search::Threads.dbStorage(),
                                                            libStream,
                                                            options.rule,
                                                            board ? board->size() : 15);
        auto          endTime    = now();
        MESSAGEL("Imported " << writeCount << " records from lib file using "
                             << (endTime - startTime) << " ms.");
    }
}

void databaseToLib()
{
    auto libPath = readPathFromInput();
    if (Search::Threads.dbStorage()) {
        MESSAGEL("Exporting to lib file " << pathToConsoleString(libPath)
                                          << ", this might take a while...");
        auto          startTime = now();
        std::ofstream libStream(libPath, std::ios::binary);
        if (!libStream) {
            ERRORL("Unable to open file " << pathToConsoleString(libPath) << " for writing.");
            return;
        }

        DBClient dbClient(*Search::Threads.dbStorage(), RECORD_MASK_ALL);
        size_t   nodeCount = ::Database::exportDatabaseToLib(dbClient, libStream, *board, options.rule);
        auto     endTime   = now();
        MESSAGEL("Exported " << nodeCount << " nodes to lib file using "
                             << (endTime - startTime) << " ms.");
    }
}

void restart()
{
    board->newGame(options.rule);
    Search::Threads.clear(false);
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
        auto candidateRange = candRange.value_or(Config::DefaultCandidateRange);
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

        if (coordStr == "DONE")
            break;

        ss.clear();
        ss << coordStr;
        ss >> x >> comma >> y;

        Pos pos = inputCoordConvert(x, y, board->size());
        if (!board->isInBoard(pos) || pos == Pos::PASS)
            ERRORL("Block coord is a pass or invalid.");

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
            if (board->checkForbiddenPoint(pos))
                std::cout << std::setfill('0') << std::setw(2)
                          << outputCoordXConvert(pos, board->size()) << std::setfill('0')
                          << std::setw(2) << outputCoordYConvert(pos, board->size());
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

void getDatabasePosition()
{
    board->newGame(options.rule);

    // Read position sequence
    while (true) {
        std::string coordStr;
        std::cin >> coordStr;
        upperInplace(coordStr);

        if (coordStr == "DONE")
            break;

        std::stringstream ss;
        ss << coordStr;
        auto pos = parseLegalCoord(ss, *board);

        if (pos.has_value())
            board->move(options.rule, *pos);
    }
}

void queryDatabaseAll(bool getPosition)
{
    using namespace ::Database;
    if (getPosition)
        getDatabasePosition();

    if (Search::Threads.dbStorage()) {
        MESSAGEL("DATABASE REFRESH");

        DBClient dbClient(*Search::Threads.dbStorage(), RECORD_MASK_ALL);

        // Get all child records and child board texts
        auto childRecords     = dbClient.queryChildren(*board, options.rule);
        auto boardPosAndTexts = dbClient.queryBoardTexts(*board, options.rule);

        // Iterate all empty positions
        auto childRecordIt      = childRecords.begin();
        auto boardPosAndTextsIt = boardPosAndTexts.begin();
        FOR_EVERY_EMPTY_POS(board, pos)
        {
            bool        printThisPos      = false;
            int         displayLabelValue = -1;
            std::string boardTextUTF8;
            DBRecord    record {};  // init as null record

            if (childRecordIt != childRecords.end() && childRecordIt->first == pos) {
                record                   = childRecordIt->second;
                std::string displayLabel = record.displayLabel();
                if (!displayLabel.empty()) {
                    // Encode the display label (maximum 4 chars) as an int32
                    if (displayLabel.length() > 4)
                        displayLabel = displayLabel.substr(0, 4);
                    displayLabelValue = 0;
                    for (char c : displayLabel)
                        displayLabelValue = (displayLabelValue << 8) | c;
                }
                childRecordIt++;
                printThisPos = true;
            }

            if (boardPosAndTextsIt != boardPosAndTexts.end() && boardPosAndTextsIt->first == pos) {
                boardTextUTF8 = std::move(boardPosAndTextsIt->second);
                boardPosAndTextsIt++;
                printThisPos = true;
            }

            // Print this position if it has DBRecord or board text
            if (printThisPos)
                MESSAGEL("DATABASE " << outputCoordXConvert(pos, board->size()) << ' '
                                     << outputCoordYConvert(pos, board->size()) << ' '
                                     << displayLabelValue << ' ' << record.value << ' '
                                     << record.depth() << ' ' << int(record.bound()) << ' '
                                     << int(!record.comment().empty()) << ' '
                                     << UTF8ToConsoleCP(boardTextUTF8));
        }

        MESSAGEL("DATABASE DONE");

        // Insert a new none record if no record is found at this position
        if (!Config::DatabaseReadonlyMode) {
            DBRecord record;
            if (board->ply() > 0 && !dbClient.query(*board, options.rule, record))
                dbClient.save(*board, options.rule, DBRecord {LABEL_NONE}, OverwriteRule::Disabled);
        }
    }
}

void queryDatabaseOne(bool getPosition)
{
    using namespace ::Database;
    if (getPosition)
        getDatabasePosition();

    if (Search::Threads.dbStorage()) {
        DBClient dbClient(*Search::Threads.dbStorage(), RECORD_MASK_ALL);
        DBRecord record;
        if (dbClient.query(*board, options.rule, record))
            MESSAGEL("DATABASE ONE " << int(record.label) << ' ' << record.value << ' '
                                     << record.depth() << ' ' << int(record.bound()) << ' '
                                     << record.displayLabel());
        else
            MESSAGEL("DATABASE ONE 0 0 0 0");
    }
}

void queryDatabaseText(bool getPosition)
{
    using namespace ::Database;
    if (getPosition)
        getDatabasePosition();

    if (Search::Threads.dbStorage()) {
        DBClient dbClient(*Search::Threads.dbStorage(), RECORD_MASK_ALL);
        DBRecord record;
        if (dbClient.query(*board, options.rule, record) && !record.isNull())
            MESSAGEL("DATABASE TEXT " << std::quoted(UTF8ToConsoleCP(record.comment())));
        else
            MESSAGEL("DATABASE TEXT \"\"");
    }
}

void editDatabaseTVD()
{
    using namespace ::Database;
    int updateMask, newLabel, newValue, newDepth;
    std::cin >> updateMask >> newLabel >> newValue >> newDepth;
    getDatabasePosition();

    if (Search::Threads.dbStorage() && !Config::DatabaseReadonlyMode) {
        switch (newLabel) {
        case 'W': newLabel = LABEL_WIN; break;
        case 'L': newLabel = LABEL_LOSE; break;
        case 'D': newLabel = LABEL_DRAW; break;
        case 'X': newLabel = LABEL_BLOCKMOVE; break;
        default: break;
        }

        DBClient dbClient(*Search::Threads.dbStorage(), (DBRecordMask)updateMask);
        DBRecord record {DBLabel(newLabel), DBValue(newValue)};
        record.setDepthBound(newDepth, BOUND_EXACT);
        dbClient.save(*board, options.rule, record, OverwriteRule::Always);
        dbClient.sync();
        queryDatabaseOne(false);
    }
}

void editDatabaseText()
{
    using namespace ::Database;
    std::string newText;
    std::cin >> std::ws >> std::quoted(newText);
    getDatabasePosition();

    if (Search::Threads.dbStorage() && !Config::DatabaseReadonlyMode) {
        DBClient dbClient(*Search::Threads.dbStorage(), RECORD_MASK_TEXT);
        DBRecord record;
        if (!dbClient.query(*board, options.rule, record))
            record = DBRecord {LABEL_NONE};

        record.setComment(ConsoleCPToUTF8(newText));
        dbClient.save(*board, options.rule, record, OverwriteRule::Always);
    }
}

void editDatabaseBoardLabel()
{
    using namespace ::Database;
    std::string coordStr, newText;
    std::cin >> coordStr;
    std::cin.ignore(1);
    std::getline(std::cin, newText);
    getDatabasePosition();

    std::stringstream ss;
    ss << coordStr;
    auto pos = parseLegalCoord(ss, *board);
    if (!pos.has_value())
        return;

    if (Search::Threads.dbStorage() && !Config::DatabaseReadonlyMode) {
        DBClient dbClient(*Search::Threads.dbStorage(), RECORD_MASK_TEXT);
        dbClient.setBoardText(*board, options.rule, *pos, ConsoleCPToUTF8(newText));
    }
}

void deleteDatabaseAll(bool getPosition)
{
    using namespace ::Database;

    std::cin >> std::ws;
    std::function<DBClient::DelType(DBRecord &)> deleteFilter = nullptr;
    if (auto firstChar = std::toupper(std::cin.peek());
        firstChar == 'W' || firstChar == 'L' || firstChar == 'N') {
        std::string deleteType;
        std::cin >> deleteType;
        upperInplace(deleteType);
        if (deleteType == "NONWL")
            deleteFilter = [](DBRecord &record) {
                return record.label == LABEL_WIN || record.label == LABEL_LOSE
                           ? DBClient::DelType::NoDelete
                           : DBClient::DelType::DeleteRecursive;
            };
        else if (deleteType == "NONWLRECURSIVE")
            deleteFilter = [](DBRecord &record) {
                return record.label == LABEL_WIN || record.label == LABEL_LOSE
                           ? DBClient::DelType::NoDeleteRecursive
                           : DBClient::DelType::DeleteRecursive;
            };
        else if (deleteType == "NONWLD")
            deleteFilter = [](DBRecord &record) {
                return isDeterminedLabel(record.label) ? DBClient::DelType::NoDelete
                                                       : DBClient::DelType::DeleteRecursive;
            };
        else if (deleteType == "NONWLDRECURSIVE")
            deleteFilter = [](DBRecord &record) {
                return isDeterminedLabel(record.label) ? DBClient::DelType::NoDeleteRecursive
                                                       : DBClient::DelType::DeleteRecursive;
            };
        else if (deleteType == "W")
            deleteFilter = [](DBRecord &record) {
                return record.label == LABEL_WIN ? DBClient::DelType::DeleteRecursive
                                                 : DBClient::DelType::NoDelete;
            };
        else if (deleteType == "WRECURSIVE")
            deleteFilter = [](DBRecord &record) {
                return record.label == LABEL_WIN ? DBClient::DelType::DeleteRecursive
                                                 : DBClient::DelType::NoDeleteRecursive;
            };
        else if (deleteType == "L")
            deleteFilter = [](DBRecord &record) {
                return record.label == LABEL_LOSE ? DBClient::DelType::DeleteRecursive
                                                  : DBClient::DelType::NoDelete;
            };
        else if (deleteType == "LRECURSIVE")
            deleteFilter = [](DBRecord &record) {
                return record.label == LABEL_LOSE ? DBClient::DelType::DeleteRecursive
                                                  : DBClient::DelType::NoDeleteRecursive;
            };
        else if (deleteType == "WL")
            deleteFilter = [](DBRecord &record) {
                return record.label == LABEL_WIN || record.label == LABEL_LOSE
                           ? DBClient::DelType::DeleteRecursive
                           : DBClient::DelType::NoDelete;
            };
        else if (deleteType == "WLRECURSIVE")
            deleteFilter = [](DBRecord &record) {
                return record.label == LABEL_WIN || record.label == LABEL_LOSE
                           ? DBClient::DelType::DeleteRecursive
                           : DBClient::DelType::NoDeleteRecursive;
            };
        else if (deleteType == "WLD")
            deleteFilter = [](DBRecord &record) {
                return isDeterminedLabel(record.label) ? DBClient::DelType::DeleteRecursive
                                                       : DBClient::DelType::NoDelete;
            };
        else if (deleteType == "WLDRECURSIVE")
            deleteFilter = [](DBRecord &record) {
                return isDeterminedLabel(record.label) ? DBClient::DelType::DeleteRecursive
                                                       : DBClient::DelType::NoDeleteRecursive;
            };
        else if (deleteType == "WLNOSTEP")
            deleteFilter = [](DBRecord &record) {
                Value value = Value(-record.value);
                return record.label == LABEL_WIN && value <= VALUE_MATE_IN_MAX_PLY
                               || record.label == LABEL_LOSE && value >= VALUE_MATED_IN_MAX_PLY
                           ? DBClient::DelType::DeleteRecursive
                           : DBClient::DelType::NoDelete;
            };
        else if (deleteType == "WLNOSTEPRECURSIVE")
            deleteFilter = [](DBRecord &record) {
                Value value = Value(-record.value);
                return record.label == LABEL_WIN && value <= VALUE_MATE_IN_MAX_PLY
                               || record.label == LABEL_LOSE && value >= VALUE_MATED_IN_MAX_PLY
                           ? DBClient::DelType::DeleteRecursive
                           : DBClient::DelType::NoDeleteRecursive;
            };
        else if (deleteType == "WLINSTEP") {
            int step;
            std::cin >> step;
            deleteFilter = [step](DBRecord &record) {
                Value value = Value(-record.value);
                return (record.label == LABEL_WIN && value > VALUE_MATE_IN_MAX_PLY
                        || record.label == LABEL_LOSE && value < VALUE_MATED_IN_MAX_PLY)
                               && mate_step(value, -1) <= step
                           ? DBClient::DelType::DeleteRecursive
                           : DBClient::DelType::NoDelete;
            };
        }
        else if (deleteType == "WLINSTEPRECURSIVE") {
            int step;
            std::cin >> step;
            deleteFilter = [step](DBRecord &record) {
                Value value = Value(-record.value);
                return (record.label == LABEL_WIN && value > VALUE_MATE_IN_MAX_PLY
                        || record.label == LABEL_LOSE && value < VALUE_MATED_IN_MAX_PLY)
                               && mate_step(value, -1) <= step
                           ? DBClient::DelType::DeleteRecursive
                           : DBClient::DelType::NoDeleteRecursive;
            };
        }
        else
            ERRORL("Unknown database delete type " << deleteType);
    }

    if (getPosition)
        getDatabasePosition();

    if (Search::Threads.dbStorage() && !Config::DatabaseReadonlyMode) {
        MESSAGEL("Deleting child records, this might take a while...");
        auto   startTime        = now();
        size_t sizeBeforeDelete = Search::Threads.dbStorage()->size();
        {
            DBClient dbClient(*Search::Threads.dbStorage(), RECORD_MASK_ALL);
            dbClient.delChildren(*board, options.rule, deleteFilter);
        }
        size_t sizeAfterDelete = Search::Threads.dbStorage()->size();
        auto   endTime         = now();
        MESSAGEL("Done deleting " << (sizeBeforeDelete - sizeAfterDelete) << " records using "
                                  << (endTime - startTime) << " ms.");

        queryDatabaseAll(false);
    }
}

void deleteDatabaseOne(bool getPosition)
{
    using namespace ::Database;
    if (getPosition)
        getDatabasePosition();

    if (Search::Threads.dbStorage() && !Config::DatabaseReadonlyMode) {
        DBClient dbClient(*Search::Threads.dbStorage(), RECORD_MASK_ALL);
        dbClient.del(*board, options.rule);
    }
}

void searchDefend()
{
    options.multiPV = board->movesLeft();
    think(*board, std::max<uint16_t>(options.multiPV, 1));
}

void splitDatabase()
{
    auto databasePath = readPathFromInput();
    if (Search::Threads.dbStorage()) {
        std::string dbPathUTF8 = databasePath.u8string();
        if (auto dbToSplit = Config::createDefaultDBStorage(dbPathUTF8)) {
            auto   startTime  = now();
            size_t writeCount = ::Database::splitDatabase(*Search::Threads.dbStorage(),
                                                          *dbToSplit,
                                                          *board,
                                                          options.rule);
            auto   endTime    = now();
            MESSAGEL("Write " << writeCount << " records into the spilted database using "
                              << (endTime - startTime) << " ms.");
        }
    }
}

void mergeDatabase()
{
    auto databasePath = readPathFromInput();
    if (Search::Threads.dbStorage()) {
        std::string dbPathUTF8 = databasePath.u8string();
        if (auto dbToMerge = Config::createDefaultDBStorage(dbPathUTF8)) {
            size_t writeCount = mergeDatabase(*Search::Threads.dbStorage(),
                                              *dbToMerge,
                                              Config::DatabaseOverwriteRule);
            MESSAGEL("Merged " << writeCount << " out of " << dbToMerge->size()
                               << " records into the database.");
        }
    }
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

        std::cout << outputCoordXConvert(opening[0], board->size()) << ','
                  << outputCoordYConvert(opening[0], board->size()) << ' '
                  << outputCoordXConvert(opening[1], board->size()) << ','
                  << outputCoordYConvert(opening[1], board->size()) << ' '
                  << outputCoordXConvert(opening[2], board->size()) << ','
                  << outputCoordYConvert(opening[2], board->size()) << std::endl;
    }
    else {
        think(*board, 1, Search::SearchOptions::BALANCE_NONE, true);
    }
}

void traceBoard()
{
    Search::Threads.waitForIdle();
    Search::Threads.main()->searchOptions = options;
    Search::Threads.main()->setBoardAndEvaluator(*board);

    std::string traceInfo  = Search::Threads.main()->board->trace();
    auto        traceLines = split(traceInfo, "\n");
    for (const auto &line : traceLines) {
        MESSAGEL(line);
    }
}

void traceSearch()
{
    Search::Threads.waitForIdle();
    Search::Threads.main()->searchOptions = options;
    Search::Threads.main()->setBoardAndEvaluator(*board);
    Board &board = *Search::Threads.main()->board;

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
                                    << (Config::valueToWinRate(evalPOB) * 100.0f) << ", SF "
                                    << std::setprecision(2) << Config::ScalingFactor << ")");

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
    if (cmd == "END")                          return Search::Threads.stopThinking(), true;
    else if (cmd == "STOP" || cmd == "YXSTOP") return Search::Threads.stopThinking(), false;
    else if (thinking)                         return false;

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
     && cmd != "YXQUERYDATABASEALLT")      Search::Threads.stopThinking();

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
    else if (cmd == "YXSAVEDATABASE")      saveDatabase();
    else if (cmd == "YXDBTOTXTALL")        databaseToTxt(false);
    else if (cmd == "YXDBTOTXT")           databaseToTxt(true);
    else if (cmd == "YXLIBTODB")           libToDatabase();
    else if (cmd == "YXDBTOLIB")           CheckBoardOK(databaseToLib);
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
    else if (cmd == "YXQUERYDATABASEALL")  CheckBoardOK([] { queryDatabaseAll(true); });
    else if (cmd == "YXQUERYDATABASEONE")  CheckBoardOK([] { queryDatabaseOne(true); });
    else if (cmd == "YXQUERYDATABASETEXT") CheckBoardOK([] { queryDatabaseText(true); });
    else if (cmd == "YXQUERYDATABASEALLT") CheckBoardOK([] { queryDatabaseAll(true); queryDatabaseText(false); });
    else if (cmd == "YXEDITTVDDATABASE")   CheckBoardOK(editDatabaseTVD);
    else if (cmd == "YXEDITTEXTDATABASE")  CheckBoardOK(editDatabaseText);
    else if (cmd == "YXEDITLABELDATABASE") CheckBoardOK(editDatabaseBoardLabel);
    else if (cmd == "YXDELETEDATABASEONE") CheckBoardOK([] { deleteDatabaseOne(true); });
    else if (cmd == "YXDELETEDATABASEALL") CheckBoardOK([] { deleteDatabaseAll(true); });
    else if (cmd == "YXSEARCHDEFEND")      CheckBoardOK(searchDefend);
    else if (cmd == "YXDBSPLIT")           CheckBoardOK(splitDatabase);
    else if (cmd == "YXDBMERGE")           CheckBoardOK(mergeDatabase);
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

/// Warp around runProtocol(), looping until exit condition is met.
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

    // If there is any thread still running, wait until they exited.
    Search::Threads.waitForIdle();
    Search::Threads.setNumThreads(0);

#ifdef __EMSCRIPTEN__
    emscripten_force_exit(EXIT_SUCCESS);
#endif
}
